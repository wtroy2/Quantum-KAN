#define EIGEN_DONT_PARALLELIZE
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <symengine/basic.h>
#include <symengine/symbol.h>
#include <symengine/add.h>
#include <symengine/mul.h>
#include <symengine/pow.h>
#include <symengine/functions.h>
#include <symengine/printers.h>
#include <symengine/complex.h>
#include <symengine/real_double.h>
#include <symengine/integer.h>
#include <symengine/dict.h>
#include <symengine/visitor.h>
#include <symengine/rational.h>
#include <symengine/parser.h> // for parse function

#include <unordered_map>
#include <vector>
#include <string>
#include <iostream>
#include <thread>
#include <numeric> // For std::accumulate
#include <cmath> // for std::abs
#include <map>
#include <omp.h>
// #include <chrono> // for getting timing values
#include <Eigen/Dense>
#include <vector>
#include <fstream>
#include "include/json.hpp"
#include <regex>


namespace py = pybind11;
using namespace SymEngine;
using namespace std;
using Eigen::ArrayXd;
using json = nlohmann::json;


// Define custom hash function for SymEngine::RCP
namespace std {
    template <>
    struct hash<SymEngine::RCP<const SymEngine::Basic>> {
        std::size_t operator()(const SymEngine::RCP<const SymEngine::Basic>& k) const {
            return std::hash<std::string>()(k->__str__());
        }
    };
}

// Define custom hash function for vector
struct vector_hash {
    template <class T>
    std::size_t operator()(const std::vector<T>& v) const {
        std::size_t seed = v.size();
        for (auto& i : v) {
            seed ^= std::hash<T>()(i) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

// Custom hash function for RCP<const Basic>
struct BasicHash {
    size_t operator()(const RCP<const Basic>& k) const {
        return std::hash<std::string>()(k->__str__());
    }
};

// Custom equality function for RCP<const Basic>
struct BasicEqual {
    bool operator()(const RCP<const Basic>& lhs, const RCP<const Basic>& rhs) const {
        return eq(*lhs, *rhs);
    }
};


struct ExpressionHash {
    std::size_t operator()(const RCP<const Basic>& expr) const {
        return expr->hash();
    }
};

struct ExpressionEqual {
    bool operator()(const RCP<const Basic>& lhs, const RCP<const Basic>& rhs) const {
        return eq(*lhs, *rhs);
    }
};

// Define Binary Variable Class
class Binary : public Symbol {
public:
    using Symbol::Symbol;

    // Manually handle the power operation
    RCP<const Basic> power(const RCP<const Basic>& other) const {
        if (is_a<Integer>(*other) && (rcp_static_cast<const Integer>(other))->is_positive()) {
            return rcp_from_this_cast<const Basic>();
        }
        return pow(rcp_from_this_cast<const Basic>(), other);
    }
};

// Utility function to create binary variables
RCP<const Binary> binary(const std::string& name) {
    return make_rcp<Binary>(name);
}


RCP<const Basic> replace_binary_powers(const RCP<const Basic>& expr) {
    if (is_a<Symbol>(*expr) && (static_cast<const Symbol&>(*expr).get_name().substr(0, 1) == "P" || static_cast<const Symbol&>(*expr).get_name().substr(0, 3) == "AUX")) {
        return expr;
    }
    if (is_a<Pow>(*expr)) {
        const auto& base = static_cast<const Pow&>(*expr).get_base();
        const auto& exp = static_cast<const Pow&>(*expr).get_exp();
        if (is_a<Symbol>(*base) && (static_cast<const Symbol&>(*base).get_name().substr(0, 1) == "P" || static_cast<const Symbol&>(*base).get_name().substr(0, 3) == "AUX")) {
            return base;
        }
    }
    if (is_a<Add>(*expr)) {
        vec_basic new_args;
        for (const auto& arg : expr->get_args()) {
            new_args.push_back(replace_binary_powers(arg));
        }
        return add(new_args);
    }
    if (is_a<Mul>(*expr)) {
        vec_basic new_args;
        for (const auto& arg : expr->get_args()) {
            new_args.push_back(replace_binary_powers(arg));
        }
        return mul(new_args);
    }
    return expr;
}

// Function to check if a symbol is a P-variable or AUX-variable
bool is_p_or_aux_variable(const RCP<const Basic>& sym) {
    if (is_a<Symbol>(*sym)) {
        const std::string& name = static_cast<const Symbol&>(*sym).get_name();
        return name.substr(0, 1) == "P" || name.substr(0, 3) == "AUX";
    }
    return false;
}

// Function to separate sub-expressions
vector<RCP<const Basic>> separate_sub_expressions(const RCP<const Basic>& expr) {
    // Expand the expression fully, including powers
    // auto expanded_expr = expand(expr);
    auto expanded_expr = expr;
    
    // cout << "Expanded expression: " << *expanded_expr << endl;

    // Extract the terms
    vec_basic terms;
    if (is_a<Add>(*expanded_expr)) {
        terms = expanded_expr->get_args();
    } else {
        terms.push_back(expanded_expr);
    }

    // Dictionary to hold common factors
    unordered_map<vector<RCP<const Basic>>, RCP<const Basic>, vector_hash> common_factors;

    // Iterate over each term
    for (const auto& term : terms) {
        // Extract the coefficient and the symbols
        RCP<const Basic> coeff = one;
        vec_basic symbols;
        if (is_a<Mul>(*term)) {
            coeff = static_cast<const Mul&>(*term).get_coef();
            symbols = static_cast<const Mul&>(*term).get_args();
        } else {
            symbols.push_back(term);
        }

        // Separate P/AUX variables and other variables
        vector<RCP<const Basic>> p_vars;
        vector<RCP<const Basic>> other_vars;
        for (const auto& sym : symbols) {
            if (is_p_or_aux_variable(sym)) {
                p_vars.push_back(sym);
            } else {
                other_vars.push_back(sym);
            }
        }

        // Ensure other_vars is non-empty to avoid multiplication issues
        RCP<const Basic> other_product = one;
        if (!other_vars.empty()) {
            other_product = mul(other_vars);
        }

        // Group terms with common P/AUX variables
        if (common_factors.find(p_vars) == common_factors.end()) {
            common_factors[p_vars] = zero;
        }
        common_factors[p_vars] = add(common_factors[p_vars], other_product);
    }

    // Create an array of sub-expressions
    vector<RCP<const Basic>> sub_expressions;
    for (const auto& pair : common_factors) {
        RCP<const Basic> grouped_other_vars = pair.second;
        if (!pair.first.empty()) {
            grouped_other_vars = mul(mul(pair.first), grouped_other_vars);
        }
        sub_expressions.push_back(grouped_other_vars);
    }

    return sub_expressions;
}

RCP<const Basic> group_common_terms_include_aux(const RCP<const Basic>& expr) {
    // Expand the expression fully
    auto expanded_expr = expand(expr);

    // Replace powers of binary variables
    expanded_expr = replace_binary_powers(expanded_expr);

    // Dictionary to hold common factors
    std::unordered_map<std::vector<RCP<const Basic>>, vec_basic, vector_hash> common_factors;

    // Iterate over each term in the expanded expression
    if (is_a<Add>(*expanded_expr)) {
        for (const auto& term : expanded_expr->get_args()) {
            // Separate coefficients and variables
            RCP<const Basic> coeff = one;
            vec_basic variables;

            if (is_a<Mul>(*term)) {
                for (const auto& factor : term->get_args()) {
                    if (is_a_Number(*factor) || is_a<RealDouble>(*factor)) {
                        coeff = mul(coeff, factor);
                    } else {
                        variables.push_back(factor);
                    }
                }
            } else if (is_a_Number(*term) || is_a<RealDouble>(*term)) {
                coeff = term;
            } else {
                variables.push_back(term);
            }

            // Separate P/AUX variables and other variables
            std::vector<RCP<const Basic>> P_vars;
            vec_basic other_vars;
            for (const auto& var : variables) {
                if (is_a<Symbol>(*var) && (static_cast<const Symbol&>(*var).get_name().substr(0, 1) == "P" || static_cast<const Symbol&>(*var).get_name().substr(0, 3) == "AUX")) {
                    P_vars.push_back(var);
                } else {
                    other_vars.push_back(var);
                }
            }

            // Create a product of the other variables
            RCP<const Basic> other_product = other_vars.empty() ? static_cast<RCP<const Basic>>(one) : mul(other_vars);

            // Group terms with common P variables
            common_factors[P_vars].push_back(mul(coeff, other_product));
        }
    } else {
        return expr;
    }

    // Combine grouped terms
    vec_basic combined_terms;
    for (const auto& group : common_factors) {
        RCP<const Basic> sum_of_coeffs = add(group.second);
        combined_terms.push_back(mul(mul(group.first), sum_of_coeffs));
    }

    return add(combined_terms);
}

RCP<const Basic> bernstein_basis_functions_symbolic_continuous_control(const RCP<const Basic>& t, int degree, const vector<RCP<const Basic>>& control_points) {
    int n = degree;
    RCP<const Basic> combined_expr = integer(0);

    for (int i = 0; i <= n; ++i) {
        auto binomial_coeff = binomial(*integer(n), static_cast<unsigned long>(i));
        vec_basic terms = {binomial_coeff, pow(t, integer(i)), pow(sub(integer(1), t), integer(n - i))};
        auto basis_function = mul(terms);
        auto full_function = mul(basis_function, control_points[i]);
        combined_expr = add(combined_expr, full_function);
    }

    return combined_expr;
}


// Function to check if a symbol is a P-variable
bool is_p_variable(const RCP<const Basic>& sym) {
    return is_a<Symbol>(*sym) && static_cast<const Symbol&>(*sym).get_name().substr(0, 1) == "P";
}

// Function to create auxiliary variables for complex products of P variables only
RCP<const Basic> create_aux_variables_top(const RCP<const Basic>& expr, unordered_map<RCP<const Basic>, RCP<const Basic>>& aux_dict) {
    if (is_a<Symbol>(*expr) || is_a<Integer>(*expr)) {
        return expr;
    }

    if (is_a<Mul>(*expr)) {
        vec_basic factors = expr->get_args();
        vector<RCP<const Basic>> p_factors;
        vector<RCP<const Basic>> non_p_factors;

        for (const auto& factor : factors) {
            if (is_p_variable(factor)) {
                p_factors.push_back(factor);
            } else {
                non_p_factors.push_back(factor);
            }
        }

        while (p_factors.size() > 1) {
            bool found = false;
            for (size_t i = 0; i < p_factors.size(); ++i) {
                for (size_t j = i + 1; j < p_factors.size(); ++j) {
                    for (const auto& aux_pair : aux_dict) {
                        if (eq(*mul(p_factors[i], p_factors[j]), *aux_pair.second)) {
                            RCP<const Basic> new_var = aux_pair.first;
                            p_factors.erase(p_factors.begin() + j);
                            p_factors.erase(p_factors.begin() + i);
                            p_factors.push_back(new_var);
                            found = true;
                            break;
                        }
                    }
                    if (found) break;
                }
                if (found) break;
            }

            if (!found) {
                if (p_factors.size() > 1) {
                    RCP<const Binary> new_var = binary("AUX" + to_string(aux_dict.size()));
                    RCP<const Basic> aux_expr = mul(p_factors[0], p_factors[1]);
                    aux_dict[new_var] = aux_expr;
                    p_factors.erase(p_factors.begin());
                    p_factors.erase(p_factors.begin());
                    p_factors.push_back(new_var);
                }
            }
        }

        vec_basic new_factors;
        new_factors.insert(new_factors.end(), non_p_factors.begin(), non_p_factors.end());
        new_factors.insert(new_factors.end(), p_factors.begin(), p_factors.end());

        return mul(new_factors);
    }

    if (is_a<Add>(*expr)) {
        vec_basic new_args;
        for (const auto& arg : expr->get_args()) {
            new_args.push_back(create_aux_variables_top(arg, aux_dict));
        }
        return add(new_args);
    }

    return expr;
}

// Function to create auxiliary variables for complex products of P variables only
RCP<const Basic> create_aux_variables(const RCP<const Basic>& expr, unordered_map<RCP<const Basic>, RCP<const Basic>>& aux_dict) {
    if (is_a<Symbol>(*expr) || is_a<Integer>(*expr)) {
        return expr;
    }

    if (is_a<Mul>(*expr)) {
        vec_basic factors = expr->get_args();
        vector<RCP<const Basic>> p_factors;
        vector<RCP<const Basic>> non_p_factors;

        for (const auto& factor : factors) {
            if (is_p_or_aux_variable(factor)) {
                p_factors.push_back(factor);
            } else {
                non_p_factors.push_back(factor);
            }
        }

        while (p_factors.size() > 2) {
            bool found = false;
            for (size_t i = 0; i < p_factors.size(); ++i) {
                for (size_t j = i + 1; j < p_factors.size(); ++j) {
                    for (const auto& aux_pair : aux_dict) {
                        if (eq(*mul(p_factors[i], p_factors[j]), *aux_pair.second)) {
                            RCP<const Basic> new_var = aux_pair.first;
                            p_factors.erase(p_factors.begin() + j);
                            p_factors.erase(p_factors.begin() + i);
                            p_factors.push_back(new_var);
                            found = true;
                            break;
                        }
                    }
                    if (found) break;
                }
                if (found) break;
            }

            if (!found) {
                if (p_factors.size() > 2) {
                    RCP<const Binary> new_var = binary("AUX" + to_string(aux_dict.size()));
                    RCP<const Basic> aux_expr = mul(p_factors[0], p_factors[1]);
                    aux_dict[new_var] = aux_expr;
                    p_factors.erase(p_factors.begin());
                    p_factors.erase(p_factors.begin());
                    p_factors.push_back(new_var);
                }
            }
        }

        vec_basic new_factors;
        new_factors.insert(new_factors.end(), non_p_factors.begin(), non_p_factors.end());
        new_factors.insert(new_factors.end(), p_factors.begin(), p_factors.end());

        return mul(new_factors);
    }

    if (is_a<Add>(*expr)) {
        vec_basic new_args;
        for (const auto& arg : expr->get_args()) {
            new_args.push_back(create_aux_variables(arg, aux_dict));
        }
        return add(new_args);
    }

    return expr;
}

// Apply auxiliary variable creation to all sub-expressions
pair<RCP<const Basic>, unordered_map<RCP<const Basic>, RCP<const Basic>>> apply_aux_variables(const RCP<const Basic>& expr, unordered_map<RCP<const Basic>, RCP<const Basic>>& existing_aux_dict, bool isTop) {
    RCP<const Basic> new_expr;
    if (isTop) {
        new_expr = create_aux_variables_top(expr, existing_aux_dict);
    } else {
        new_expr = create_aux_variables(expr, existing_aux_dict);
    }
    return {new_expr, existing_aux_dict};
}

double find_max_coefficient(const RCP<const Basic>& expr) {
    double max_coeff = 0.0;

    // Expand the expression fully, including powers
    auto expanded_expr = expand(expr);

    // Extract the terms
    vec_basic terms;
    if (is_a<Add>(*expanded_expr)) {
        terms = expanded_expr->get_args();
    } else {
        terms.push_back(expanded_expr);
    }

    // Iterate over each term
    for (const auto& term : terms) {
        double coeff_value = 1.0; // default coefficient is 1

        if (is_a<Mul>(*term)) {
            auto coeff = static_cast<const Mul&>(*term).get_coef();
            coeff_value = eval_double(*coeff);
            // cout << "coeff_value: " << coeff_value << endl;
        } else if (is_a<RealDouble>(*term) || is_a<Integer>(*term) || is_a<Rational>(*term)) {
            coeff_value = eval_double(*term);
            // cout << "coeff_value: " << coeff_value << endl;
        }

        // Update the max coefficient if this term's coefficient is larger in absolute value
        if (std::abs(coeff_value) > std::abs(max_coeff)) {
            max_coeff = coeff_value;
        }
    }

    return std::abs(max_coeff);
}

vector<RCP<const Basic>> generate_penalty_functions(const unordered_map<RCP<const Basic>, RCP<const Basic>>& aux_dict_all, double penalty_coefficient) {
    vector<RCP<const Basic>> penalties;
    
    for (const auto& aux_pair : aux_dict_all) {
        const auto& aux_var = aux_pair.first;
        const auto& aux_expr = aux_pair.second;

        // Parse the auxiliary expression to extract variables
        vec_basic vars_in_expr;
        if (is_a<Mul>(*aux_expr)) {
            vars_in_expr = aux_expr->get_args();
        } else {
            throw std::runtime_error("Unexpected number of variables in expression");
        }
        
        if (vars_in_expr.size() != 2) {
            throw std::runtime_error("Unexpected number of variables in expression");
        }
        
        auto x = vars_in_expr[0];
        auto y = vars_in_expr[1];
        
        // Penalty terms to ensure z = x * y
        auto penalty1 = add(add(mul(real_double(penalty_coefficient), mul(x, y)), mul(integer(-2), mul(real_double(penalty_coefficient), mul(x, aux_var)))), add(mul(integer(-2), mul(real_double(penalty_coefficient), mul(y, aux_var))), mul(integer(3), mul(real_double(penalty_coefficient), aux_var))));
        
        // Add penalty terms to the list
        penalties.push_back(penalty1);
    }
    
    return penalties;
}


std::unordered_set<RCP<const Basic>, BasicHash, BasicEqual> extract_unique_xyz_terms(const RCP<const Basic>& expr) {
    std::unordered_set<RCP<const Basic>, BasicHash, BasicEqual> unique_terms;

    if (is_a<Add>(*expr)) {
        for (const auto& term : expr->get_args()) {
            vec_basic variables;
            vec_basic p_vars;

            if (is_a<Mul>(*term)) {
                for (const auto& factor : term->get_args()) {
                    if (is_a<Symbol>(*factor) && (static_cast<const Symbol&>(*factor).get_name().substr(0, 1) == "P" || 
                                                  static_cast<const Symbol&>(*factor).get_name().substr(0, 3) == "AUX")) {
                        p_vars.push_back(factor);
                    } else {
                        variables.push_back(factor);
                    }
                }
            } else {
                if (is_a<Symbol>(*term) && (static_cast<const Symbol&>(*term).get_name().substr(0, 1) == "P" || 
                                            static_cast<const Symbol&>(*term).get_name().substr(0, 3) == "AUX")) {
                    p_vars.push_back(term);
                } else {
                    variables.push_back(term);
                }
            }

            if (!variables.empty()) {
                unique_terms.insert(mul(variables));
            }
        }
    } else {
        vec_basic variables;
        vec_basic p_vars;

        if (is_a<Mul>(*expr)) {
            for (const auto& factor : expr->get_args()) {
                if (is_a<Symbol>(*factor) && (static_cast<const Symbol&>(*factor).get_name().substr(0, 1) == "P" || 
                                              static_cast<const Symbol&>(*factor).get_name().substr(0, 3) == "AUX")) {
                    p_vars.push_back(factor);
                } else {
                    variables.push_back(factor);
                }
            }
        } else {
            if (is_a<Symbol>(*expr) && (static_cast<const Symbol&>(*expr).get_name().substr(0, 1) == "P" || 
                                        static_cast<const Symbol&>(*expr).get_name().substr(0, 3) == "AUX")) {
                p_vars.push_back(expr);
            } else {
                variables.push_back(expr);
            }
        }

        if (!variables.empty()) {
            unique_terms.insert(mul(variables));
        }
    }

    return unique_terms;
}



// Function to evaluate SymEngine expression and return Eigen array
Eigen::ArrayXd evaluate_symengine_expr(const RCP<const Basic> &expr, const ArrayXd &x, const ArrayXd &y, const ArrayXd &z) {
    Eigen::ArrayXd result = Eigen::ArrayXd::Zero(x.size());

    // Evaluate the expression based on its type
    if (is_a<const Symbol>(*expr)) {
        if (eq(*expr, *symbol("x"))) {
            result = x;
        } else if (eq(*expr, *symbol("y"))) {
            result = y;
        } else if (eq(*expr, *symbol("z"))) {
            result = z;
        }
    } else if (is_a<const Add>(*expr)) {
        for (const auto &arg : expr->get_args()) {
            result += evaluate_symengine_expr(arg, x, y, z);
        }
    } else if (is_a<const Mul>(*expr)) {
        result.setOnes();
        for (const auto &arg : expr->get_args()) {
            result *= evaluate_symengine_expr(arg, x, y, z);
        }
    } else if (is_a<const Pow>(*expr)) {
        const auto &base = static_cast<const Pow &>(*expr).get_base();
        const auto &exp = static_cast<const Pow &>(*expr).get_exp();
        Eigen::ArrayXd base_eval = evaluate_symengine_expr(base, x, y, z);
        double exponent = eval_double(*exp);
        result = base_eval.pow(exponent);
    } else if (is_a<const RealDouble>(*expr) || is_a<const Integer>(*expr) || is_a<const Rational>(*expr)) {
        result = Eigen::ArrayXd::Constant(x.size(), eval_double(*expr));
    } else {
        cout << "The unsupported expression: " << *expr << endl;
        throw runtime_error("Unsupported SymEngine expression type");
    }

    return result;
}

void precompute_powers(const ArrayXd &x, int max_exp, std::vector<ArrayXd> &x_powers) {
    x_powers.resize(max_exp + 1);
    x_powers[0] = ArrayXd::Ones(x.size());
    for (int i = 1; i <= max_exp; ++i) {
        x_powers[i] = x_powers[i-1] * x;
    }
}

void precompute_powers_and_combinations(const ArrayXd &x, const ArrayXd &y, const ArrayXd &z, int max_exp,
                                        std::unordered_map<std::string, ArrayXd> &precomputed_values) {
    // Ensure x, y, z have valid sizes
    if (x.size() == 0 || y.size() == 0 || z.size() == 0) {
        throw std::invalid_argument("Input arrays must have non-zero size.");
    }
    // std::cout << "starting precompute_powers_and_combinations" << endl;
    // Precompute powers of individual variables
    for (int i = 1; i <= max_exp; ++i) {
        if (i > 1) {
            precomputed_values["x**" + std::to_string(i)] = x.pow(i);
            precomputed_values["y**" + std::to_string(i)] = y.pow(i);
            precomputed_values["z**" + std::to_string(i)] = z.pow(i);
        } else {
            precomputed_values["x"] = x.pow(i);
            precomputed_values["y"] = y.pow(i);
            precomputed_values["z"] = z.pow(i);     
        }
    }

    // Print all precomputed values
    // std::cout << "Precomputed values:" << std::endl;
    // for (const auto &pair : precomputed_values) {
    //     std::cout << pair.first << std::endl; //" = " << pair.second.transpose() << std::endl; // Print the key and corresponding ArrayXd
    // }
    // std::cout << "intial precompute_powers_and_combinations" << endl;
    std::string temp;

    // Precompute combinations like x^i * y^j * z^k where i + j + k <= max_exp
    for (int i = 0; i <= max_exp; ++i) {
        for (int j = 0; j <= max_exp - i; ++j) {
            for (int k = 0; k <= max_exp - i - j; ++k) {
                // Skip the term that is just 1 (x^0 * y^0 * z^0)
                if (i == 0 && j == 0 && k == 0) continue;
                // std::cout << i << j<< k << endl;
                std::string key;
                ArrayXd product = ArrayXd::Ones(x.size()); // Start with 1
                if (i > 0) {
                    temp = std::to_string(i);
                    if (temp != "1"){
                        key += "x**" + temp;
                        product *= precomputed_values["x**" + temp];
                    } else {
                        key += "x";
                        product *= precomputed_values["x"];
                    }
                    if (j > 0) {
                        temp = std::to_string(j);
                        if (temp != "1"){
                            key += "*y**" + temp;
                            product *= precomputed_values["y**" + temp];
                        } else {
                            key += "*y";
                            product *= precomputed_values["y"];
                        }
                    }
                    if (k > 0) {
                        temp = std::to_string(k);
                        if (temp != "1"){
                            key += "*z**" + temp;
                            product *= precomputed_values["z**" + temp];
                        } else {
                            key += "*z";
                            product *= precomputed_values["z"];
                        }
                    }
                } else if (j > 0) {
                    temp = std::to_string(j);
                    if (temp != "1"){
                        key += "y**" + temp;
                        product *= precomputed_values["y**" + temp];
                    } else {
                        key += "y";
                        product *= precomputed_values["y"];
                    }
                    if (k > 0) {
                        temp = std::to_string(k);
                        if (temp != "1"){
                            key += "*z**" + temp;
                            product *= precomputed_values["z**" + temp];
                        } else {
                            key += "*z";
                            product *= precomputed_values["z"];
                        }
                    }
                } else if (k > 0) {
                    temp = std::to_string(k);
                    if (temp != "1"){
                        key += "z**" + temp;
                        product *= precomputed_values["z**" + temp];
                    } else {
                        key += "z";
                        product *= precomputed_values["z"];
                    }
                }

                precomputed_values[key] = product;
            }
        }
    }
}

// inline ArrayXd evaluate_symengine_expr_optimized(const RCP<const Basic> &expr, const ArrayXd &x, const ArrayXd &y, const ArrayXd &z, 
//                                           const std::vector<ArrayXd> &x_powers, const std::vector<ArrayXd> &y_powers, const std::vector<ArrayXd> &z_powers,
//                                           ArrayXd &temp1, ArrayXd &temp2) {
//     if (is_a<const Symbol>(*expr)) {
//         const Symbol &sym = static_cast<const Symbol &>(*expr);
//         if (eq(sym, *symbol("x"))) {
//             return x;
//         } else if (eq(sym, *symbol("y"))) {
//             return y;
//         } else if (eq(sym, *symbol("z"))) {
//             return z;
//         }
//     } else if (is_a<const RealDouble>(*expr) || is_a<const Integer>(*expr) || is_a<const Rational>(*expr)) {
//         return ArrayXd::Constant(x.size(), eval_double(*expr));
//     } else if (is_a<const Add>(*expr)) {
//         temp1.setZero();
//         for (const auto &arg : expr->get_args()) {
//             temp1 += evaluate_symengine_expr_optimized(arg, x, y, z, x_powers, y_powers, z_powers, temp1, temp2);
//         }
//         return temp1;
//     } else if (is_a<const Mul>(*expr)) {
//         temp2.setOnes();
//         for (const auto &arg : expr->get_args()) {
//             temp2 *= evaluate_symengine_expr_optimized(arg, x, y, z, x_powers, y_powers, z_powers, temp1, temp2);
//         }
//         return temp2;
//     } else if (is_a<const Pow>(*expr)) {
//         const auto &pow_expr = static_cast<const Pow &>(*expr);
//         ArrayXd base_eval = evaluate_symengine_expr_optimized(pow_expr.get_base(), x, y, z, x_powers, y_powers, z_powers, temp1, temp2);
//         double exponent = eval_double(*pow_expr.get_exp());
//         if (eq(*pow_expr.get_base(), *symbol("x")) && exponent == static_cast<int>(exponent) && exponent >= 0 && exponent < x_powers.size()) {
//             return x_powers[static_cast<int>(exponent)];
//         } else if (eq(*pow_expr.get_base(), *symbol("y")) && exponent == static_cast<int>(exponent) && exponent >= 0 && exponent < y_powers.size()) {
//             return y_powers[static_cast<int>(exponent)];
//         } else if (eq(*pow_expr.get_base(), *symbol("z")) && exponent == static_cast<int>(exponent) && exponent >= 0 && exponent < z_powers.size()) {
//             return z_powers[static_cast<int>(exponent)];
//         } else {
//             return base_eval.pow(exponent);
//         }
//     } else {
//         throw std::runtime_error("Unsupported SymEngine expression type");
//     }
//     return ArrayXd::Zero(x.size());
// }

// inline ArrayXd evaluate_symengine_expr_optimized(
//     const RCP<const Basic> &expr,
//     const std::unordered_map<std::string, ArrayXd> &precomputed_values,
//     ArrayXd &temp1, ArrayXd &temp2) {
    
//     if (is_a<const Symbol>(*expr)) {
//         const Symbol &sym = static_cast<const Symbol &>(*expr);
//         std::string name = sym.get_name();
//         if (precomputed_values.count(name)) {
//             return precomputed_values.at(name);
//         }
//     } else if (is_a<const RealDouble>(*expr) || is_a<const Integer>(*expr) || is_a<const Rational>(*expr)) {
//         return ArrayXd::Constant(temp1.size(), eval_double(*expr));
//     } else if (is_a<const Add>(*expr)) {
//         temp1.setZero();
//         for (const auto &arg : expr->get_args()) {
//             temp1 += evaluate_symengine_expr_optimized(arg, precomputed_values, temp1, temp2);
//         }
//         return temp1;
//     } else if (is_a<const Mul>(*expr)) {
//         temp2.setOnes();
//         for (const auto &arg : expr->get_args()) {
//             temp2 *= evaluate_symengine_expr_optimized(arg, precomputed_values, temp1, temp2);
//         }
//         return temp2;
//     } else if (is_a<const Pow>(*expr)) {
//         const auto &pow_expr = static_cast<const Pow &>(*expr);
//         std::string base_str = pow_expr.get_base()->__str__();
//         double exponent = eval_double(*pow_expr.get_exp());
//         std::string key = base_str + std::to_string(static_cast<int>(exponent));

//         if (precomputed_values.count(key)) {
//             return precomputed_values.at(key);
//         } else {
//             ArrayXd base_eval = evaluate_symengine_expr_optimized(pow_expr.get_base(), precomputed_values, temp1, temp2);
//             return base_eval.pow(exponent);
//         }
//     } else {
//         throw std::runtime_error("Unsupported SymEngine expression type");
//     }

//     return ArrayXd::Zero(temp1.size());
// }

// Function to separate terms by + or - not inside parentheses or fractions
std::vector<std::string> separate_terms(const std::string &expr_str) {
    std::vector<std::string> terms;
    std::string term;
    int depth = 0;
    for (size_t i = 0; i < expr_str.size(); ++i) {
        char c = expr_str[i];
        if (c == '(') depth++;
        if (c == ')') depth--;
        if ((c == '+' || c == '-') && depth == 0) {
            if (!term.empty()) {
                terms.push_back(term);
            }
            term = c; // Start the new term with + or -
        } else {
            term += c;
        }
    }
    if (!term.empty()) {
        terms.push_back(term);
    }
    return terms;
}

// Function to parse the coefficient and base key from a term
void parse_coefficient_and_key(const std::string &term, double &coefficient, std::string &base_key) {
    coefficient = 1.0; // Default coefficient
    base_key = term;   // Default base key is the whole term

    size_t pos = term.find('*');
    std::string coef_str;

    // Check if the term starts with a variable or an operator
    if (term[0] != 'x' && term[0] != 'y' && term[0] != 'z') {
        // Look for a sign at the beginning
        bool negative = false;
        if (term[0] == '-') {
            negative = true;
            coef_str = term.substr(1, pos - 1);
        } else if (term[0] == '+') {
            coef_str = term.substr(1, pos - 1);
        } else {
            coef_str = term.substr(0, pos);
        }

        // Remove leading/trailing whitespaces
        coef_str.erase(0, coef_str.find_first_not_of(" \n\r\t"));
        coef_str.erase(coef_str.find_last_not_of(" \n\r\t")+1);

        // Handle coefficient parsing including fractions
        // std::cout << "coef_str: " << coef_str << endl;
        if (coef_str[0] == '(') {
            try {
                size_t frac_pos = coef_str.find('/');
                double numerator = std::stod(coef_str.substr(1, frac_pos));
                double denominator = std::stod(coef_str.substr(frac_pos + 1, coef_str.size() - 1));
                // std::cout << "numerator: " << numerator << endl;
                // std::cout << "denominator: " << denominator << endl;
                
                coefficient = numerator / denominator;
            } catch (const std::exception &) {
                std::cerr << "Invalid coefficient format: " << coef_str << std::endl;
                coefficient = 1.0;
            }
        } else {
            try {
                coefficient = std::stod(coef_str);
            } catch (const std::invalid_argument &) {
                cout << "Invalid term: " << term << endl;
                std::cerr << "Invalid coefficient: " << coef_str << std::endl;
                coefficient = 1.0;
            }
        }

        // Apply the negative sign if found at the start
        if (negative) {
            coefficient = -coefficient;
        }

        // The base key should not include the coefficient part
        if (pos != std::string::npos) {
            base_key = term.substr(pos + 1);
        } else {
            base_key = ""; // In case there is no '*', base_key should be empty if coefficient parsing is correct
        }
    } else {
        // If the term starts with a variable, the coefficient is implicitly 1
        coefficient = 1.0;
        base_key = term;
    }

    // Trim any leading or trailing whitespace from base_key
    base_key.erase(base_key.find_last_not_of(" \n\r\t")+1);
    // base_key.erase(0, base_key.find_first_not_of(" \n\r\t"));
}

// Function to parse the coefficient and base key from a term
// Example function to evaluate base expression (to be implemented based on your logic)
ArrayXd evaluate_base_expression(const std::string &base_key, 
                                 std::unordered_map<std::string, ArrayXd> &precomputed_values,
                                 ArrayXd &temp1, ArrayXd &temp2) {
    // Implement your logic to evaluate the base expression (e.g., "x**2*z") here
    // This function should return the computed ArrayXd
    return ArrayXd::Zero(temp1.size()); // Placeholder return
}

inline ArrayXd evaluate_symengine_expr_optimized(
    const RCP<const Basic> &expr,
    std::unordered_map<std::string, ArrayXd> &precomputed_values, // Now passed by reference for modification
    ArrayXd &temp1, ArrayXd &temp2) {

    std::string expr_str = expr->__str__();
    // std::cout << "Evaluating expression: " << expr_str << std::endl;

    ArrayXd result(temp1.size());
    result.setZero();

    // Separate the expression by + or - not inside parentheses or fractions
    std::vector<std::string> terms = separate_terms(expr_str);
    bool add_term;

    // Evaluate each term
    for (std::string term : terms) {
        double coefficient;
        std::string base_key;
        std::string coef_str;

        // std::cout << "Term: " << term << std::endl;

        // Check for leading + or - and adjust accordingly
        if (!term.empty() && term[0] == '-') {
            term.erase(0, 1); // Remove the negative sign
            add_term = false;
        } else {
            if (!term.empty() && term[0] == '+') {
                term.erase(0, 1); // Remove the positive sign
            }
            add_term = true;
        }

        // Remove leading/trailing whitespaces from the term
        term.erase(0, term.find_first_not_of(" \n\r\t"));
        term.erase(term.find_last_not_of(" \n\r\t") + 1);
        // std::cout << "End Term: " << term << std::endl;

        // Check if the base expression is in the precomputed values
        if (precomputed_values.find(term) != precomputed_values.end()) {
            if (add_term){
                result += precomputed_values[term];
            } else {
                result -= precomputed_values[term];
            }
        } else {
            // parse_coefficient_and_key(term, coefficient, base_key);

            // Debug print for each term
            // std::cout << "Term: " << term << ", Coefficient: " << coefficient << ", Base Key: " << base_key << std::endl;

            // test just to see if I use the base values


            size_t pos = term.find('*');
            // Check if the term starts with a variable or an operator
            coef_str = term.substr(0, pos);

            // Handle coefficient parsing including fractions
            // std::cout << "coef_str: " << coef_str << endl;
            if (coef_str[0] == '(') {
                try {
                    size_t frac_pos = coef_str.find('/');
                    double numerator = std::stod(coef_str.substr(1, frac_pos));
                    double denominator = std::stod(coef_str.substr(frac_pos + 1, coef_str.size() - 1));
                    // std::cout << "numerator: " << numerator << endl;
                    // std::cout << "denominator: " << denominator << endl;
                    
                    coefficient = numerator / denominator;
                } catch (const std::exception &) {
                    std::cerr << "Invalid coefficient format: " << coef_str << std::endl;
                    coefficient = 1.0;
                }
            } else {
                try {
                    coefficient = std::stod(coef_str);
                } catch (const std::invalid_argument &) {
                    // cout << "Invalid term: " << term << endl;
                    // std::cerr << "Invalid coefficient: " << coef_str << std::endl;
                    coefficient = 1.0;
                }
            }


            // The base key should not include the coefficient part
            if (pos != std::string::npos) {
                base_key = term.substr(pos + 1);
            } else {
                base_key = ""; // In case there is no '*', base_key should be empty if coefficient parsing is correct
            }
            if (precomputed_values.find(base_key) != precomputed_values.end()) {
                if (add_term){
                    result += coefficient * precomputed_values[base_key];
                } else {
                    result -= coefficient * precomputed_values[base_key];
                }
            } else {
                if (add_term){
                    result += coefficient;
                } else {
                    result -= coefficient;
                }
                // std::cerr << "Invalid term: " << term << std::endl;
            }





            // end of new stuff
            // if (precomputed_values.find(base_key) != precomputed_values.end()) {
            //     temp1 = coefficient * precomputed_values[base_key];
            // } else {
            //     // Handle the case where the base_key is not found in precomputed_values
            //     temp1 = ArrayXd::Zero(temp1.size()); // Replace with proper evaluation if needed
            // }

            // if (add_term) {
            //     result += temp1;
            // } else {
            //     result -= temp1;
            // }

            // precomputed_values[term] = temp1;
        }
    }

    // Print all precomputed values
    // std::cout << "Precomputed values:" << std::endl;
    // for (const auto &pair : precomputed_values) {
    //     std::cout << pair.first << std::endl; //" = " << pair.second.transpose() << std::endl; // Print the key and corresponding ArrayXd
    // }

    return result;
}


// Function to evaluate all unique xyz expressions for a dataset
void evaluate_unique_xyz_expressions(
    const unordered_map<RCP<const Basic>, vector<RCP<const Basic>>, BasicHash, BasicEqual> &xyz_to_pvars,
    const Eigen::ArrayXd &x_eigen,
    const Eigen::ArrayXd &y_eigen,
    const Eigen::ArrayXd &z_eigen,
    unordered_map<RCP<const Basic>, Eigen::ArrayXd, BasicHash, BasicEqual> &evaluated_xyz_expressions
) {
    std::vector<ArrayXd> x_powers, y_powers, z_powers;
    for (const auto &pair : xyz_to_pvars) {
        RCP<const Basic> xyz_expr = pair.first;
        // cout << "Evaluating expression: " << *xyz_expr << " = " << endl;
        evaluated_xyz_expressions[xyz_expr] = evaluate_symengine_expr(xyz_expr, x_eigen, y_eigen, z_eigen);
    }
}

void evaluate_unique_xyz_expressions_optimized(
    const unordered_map<RCP<const Basic>, vector<RCP<const Basic>>, BasicHash, BasicEqual> &xyz_to_pvars,
    const Eigen::ArrayXd &x_eigen,
    const Eigen::ArrayXd &y_eigen,
    const Eigen::ArrayXd &z_eigen,
    unordered_map<RCP<const Basic>, Eigen::ArrayXd, BasicHash, BasicEqual> &evaluated_xyz_expressions, int max_exp
) {
    // std::vector<ArrayXd> x_powers, y_powers, z_powers;
    std::unordered_map<std::string, ArrayXd> precomputed_values;
    auto start_precompute_powers = std::chrono::high_resolution_clock::now(); // End total timing

    precompute_powers_and_combinations(x_eigen, y_eigen, z_eigen, max_exp, precomputed_values);
    auto end_precompute_powers = std::chrono::high_resolution_clock::now(); // End total timing

    // Calculate and print elapsed times
    std::chrono::duration<double> elapsed_precompute_powers = end_precompute_powers - start_precompute_powers;
    cout << "Time taken for elapsed_precompute_powers: " << elapsed_precompute_powers.count() << " seconds" << endl;

    // precompute_powers(x_eigen, max_exp, x_powers);
    // precompute_powers(y_eigen, max_exp, y_powers);
    // precompute_powers(z_eigen, max_exp, z_powers);

    // Print all precomputed values
    // std::cout << "Precomputed values:" << std::endl;
    // for (const auto &pair : precomputed_values) {
    //     std::cout << pair.first << std::endl; //" = " << pair.second.transpose() << std::endl; // Print the key and corresponding ArrayXd
    // }

    ArrayXd temp1(x_eigen.size());
    ArrayXd temp2(x_eigen.size());
    for (const auto &pair : xyz_to_pvars) {
        RCP<const Basic> xyz_expr = pair.first;
        // cout << "Evaluating expression: " << *xyz_expr << " = " << endl;
        evaluated_xyz_expressions[xyz_expr] = evaluate_symengine_expr_optimized(xyz_expr, precomputed_values, temp1, temp2);
        // evaluated_xyz_expressions[xyz_expr] = evaluate_symengine_expr_optimized(xyz_expr, x_eigen, y_eigen, z_eigen, x_powers, y_powers, z_powers, temp1, temp2);
    }
}

unordered_map<RCP<const Basic>, vector<RCP<const Basic>>, BasicHash, BasicEqual> map_xyz_to_pvars(const vector<RCP<const Basic>>& sub_expressions) {
    unordered_map<RCP<const Basic>, vector<RCP<const Basic>>, BasicHash, BasicEqual> xyz_to_pvars;

    for (const auto& expr : sub_expressions) {
        vec_basic xyz_terms;
        vec_basic p_var_terms;

        if (is_a<Mul>(*expr)) {
            for (const auto& factor : expr->get_args()) {
                if (is_p_or_aux_variable(factor)) {
                    p_var_terms.push_back(factor);
                } else {
                    xyz_terms.push_back(factor);
                }
            }
        } else if (is_p_or_aux_variable(expr)) {
            p_var_terms.push_back(expr);
        } else {
            xyz_terms.push_back(expr);
        }

        RCP<const Basic> xyz_expr = xyz_terms.empty() ? expr : mul(xyz_terms);
        RCP<const Basic> p_var_expr = p_var_terms.empty() ? static_cast<RCP<const Basic>>(one) : mul(p_var_terms);

        xyz_to_pvars[xyz_expr].push_back(p_var_expr);
    }

    return xyz_to_pvars;
}

// Function to evaluate and combine expressions for a dataset
RCP<const Basic> evaluate_and_combine(
    const unordered_map<RCP<const Basic>, vector<RCP<const Basic>>, BasicHash, BasicEqual> &xyz_to_pvars,
    const unordered_map<RCP<const Basic>, Eigen::ArrayXd, BasicHash, BasicEqual> &evaluated_xyz_expressions
) {
    RCP<const Basic> final_result = zero;

    for (const auto &pair : xyz_to_pvars) {
        RCP<const Basic> xyz_expr = pair.first;
        const auto &p_vars = pair.second;
        const Eigen::ArrayXd &evaluated_values = evaluated_xyz_expressions.at(xyz_expr);

        double sum = evaluated_values.sum();
        for (const auto &p_var_expr : p_vars) {
            final_result = add(final_result, mul(real_double(sum), p_var_expr));
        }
    }

    return final_result;
}

// Helper function to check if an expression contains a specific variable
bool contains_variable(const RCP<const Basic>& expr, const RCP<const Basic>& var) {
    if (eq(*expr, *var)) {
        return true;
    } else if (is_a<Add>(*expr) || is_a<Mul>(*expr)) {
        for (const auto& arg : expr->get_args()) {
            if (contains_variable(arg, var)) {
                return true;
            }
        }
    } else if (is_a<Pow>(*expr)) {
        const auto& base = static_cast<const Pow&>(*expr).get_base();
        const auto& exp = static_cast<const Pow&>(*expr).get_exp();
        if (contains_variable(base, var) || contains_variable(exp, var)) {
            return true;
        }
    }
    return false;
}

void filter_aux_dict(
    const RCP<const Basic>& aux_all_sub_expressions_equation,
    unordered_map<RCP<const Basic>, RCP<const Basic>>& aux_dict_final) {
    // Create a new map to store the filtered auxiliary variables
    unordered_map<RCP<const Basic>, RCP<const Basic>> filtered_aux_dict;

    // Iterate through each auxiliary variable in aux_dict_final
    for (const auto& aux_pair : aux_dict_final) {
        const auto& aux_var = aux_pair.first;

        // Check if aux_var is present in aux_all_sub_expressions_equation
        if (contains_variable(aux_all_sub_expressions_equation, aux_var)) {
            filtered_aux_dict[aux_var] = aux_pair.second;
        }
    }

    // Replace the original dictionary with the filtered one
    aux_dict_final = std::move(filtered_aux_dict);
}

// Save function
// Function to save a JSON object to a file
void save_json_to_file(const json& j, const std::string& filename) {
    std::ofstream file(filename);
    if (file.is_open()) {
        file << j.dump(4); // Pretty print with 4 spaces
        file.close();
    } else {
        throw std::runtime_error("Unable to open file for writing: " + filename);
    }
}

// Function to load a JSON object from a file
json load_json_from_file(const std::string& filename) {
    std::ifstream file(filename);
    if (file.is_open()) {
        json j;
        file >> j;
        file.close();
        return j;
    } else {
        throw std::runtime_error("Unable to open file for reading: " + filename);
    }
}

// Function to save data to JSON files
void save_data(
    const RCP<const Basic>& symbolic_sum,
    const RCP<const Basic>& aux_all_sub_expressions_equation,
    const unordered_map<RCP<const Basic>, RCP<const Basic>>& aux_dict_final,
    const vector<vector<RCP<const Basic>>>& coefficients_plus1,
    const vector<vector<RCP<const Basic>>>& coefficients_minus1,
    const vector<vector<RCP<const Basic>>>& coefficients_plus2,
    const vector<vector<RCP<const Basic>>>& coefficients_minus2,
    const string& filename
) {
    json j;
    j["symbolic_sum"] = symbolic_sum->__str__();
    j["aux_all_sub_expressions_equation"] = aux_all_sub_expressions_equation->__str__();

    json aux_dict_json;
    for (const auto& pair : aux_dict_final) {
        aux_dict_json[pair.first->__str__()] = pair.second->__str__();
    }
    j["aux_dict_final"] = aux_dict_json;

    auto convert_coefficients_to_json = [](const vector<vector<RCP<const Basic>>>& coefficients) {
        json coeffs_json;
        for (size_t i = 0; i < coefficients.size(); ++i) {
            for (size_t j = 0; j < coefficients[i].size(); ++j) {
                coeffs_json[to_string(i)][to_string(j)] = coefficients[i][j]->__str__();
            }
        }
        return coeffs_json;
    };

    j["coefficients_plus1"] = convert_coefficients_to_json(coefficients_plus1);
    j["coefficients_minus1"] = convert_coefficients_to_json(coefficients_minus1);
    j["coefficients_plus2"] = convert_coefficients_to_json(coefficients_plus2);
    j["coefficients_minus2"] = convert_coefficients_to_json(coefficients_minus2);

    save_json_to_file(j, filename);
}

void save_data_2_layer(
    const RCP<const Basic>& symbolic_sum_no_mean,
    int x_data_size,
    const RCP<const Basic>& aux_all_sub_expressions_equation,
    const unordered_map<RCP<const Basic>, RCP<const Basic>>& aux_dict_final,
    const vector<vector<RCP<const Basic>>>& coefficients_plus1,
    const vector<vector<RCP<const Basic>>>& coefficients_plus2,
    const vector<vector<RCP<const Basic>>>& coefficients_plus3,
    const string& filename
) {
    json j;
    j["symbolic_sum_no_mean"] = symbolic_sum_no_mean->__str__();
    j["x_data_size"] = x_data_size;
    j["aux_all_sub_expressions_equation"] = aux_all_sub_expressions_equation->__str__();

    json aux_dict_json;
    for (const auto& pair : aux_dict_final) {
        aux_dict_json[pair.first->__str__()] = pair.second->__str__();
    }
    j["aux_dict_final"] = aux_dict_json;

    auto convert_coefficients_to_json = [](const vector<vector<RCP<const Basic>>>& coefficients) {
        json coeffs_json;
        for (size_t i = 0; i < coefficients.size(); ++i) {
            for (size_t j = 0; j < coefficients[i].size(); ++j) {
                coeffs_json[to_string(i)][to_string(j)] = coefficients[i][j]->__str__();
            }
        }
        return coeffs_json;
    };

    j["coefficients_plus1"] = convert_coefficients_to_json(coefficients_plus1);
    j["coefficients_plus2"] = convert_coefficients_to_json(coefficients_plus2);
    j["coefficients_plus3"] = convert_coefficients_to_json(coefficients_plus3);

    save_json_to_file(j, filename);
}

// Function to load data from JSON files
void load_data(
    RCP<const Basic>& symbolic_sum,
    RCP<const Basic>& aux_all_sub_expressions_equation,
    unordered_map<RCP<const Basic>, RCP<const Basic>>& aux_dict_final,
    vector<vector<RCP<const Basic>>>& coefficients_plus1,
    vector<vector<RCP<const Basic>>>& coefficients_minus1,
    vector<vector<RCP<const Basic>>>& coefficients_plus2,
    vector<vector<RCP<const Basic>>>& coefficients_minus2,
    const string& filename
) {
    json j = load_json_from_file(filename);

    symbolic_sum = SymEngine::parse(j["symbolic_sum"].get<string>());
    aux_all_sub_expressions_equation = SymEngine::parse(j["aux_all_sub_expressions_equation"].get<string>());

    for (const auto& pair : j["aux_dict_final"].items()) {
        aux_dict_final[SymEngine::parse(pair.key())] = SymEngine::parse(pair.value().get<string>());
    }

    auto convert_json_to_coefficients = [](const json& coeffs_json) {
        vector<vector<RCP<const Basic>>> coefficients;
        for (const auto& row : coeffs_json.items()) {
            vector<RCP<const Basic>> coeff_row;
            for (const auto& coeff : row.value().items()) {
                coeff_row.push_back(SymEngine::parse(coeff.value().get<string>()));
            }
            coefficients.push_back(coeff_row);
        }
        return coefficients;
    };

    coefficients_plus1 = convert_json_to_coefficients(j["coefficients_plus1"]);
    coefficients_minus1 = convert_json_to_coefficients(j["coefficients_minus1"]);
    coefficients_plus2 = convert_json_to_coefficients(j["coefficients_plus2"]);
    coefficients_minus2 = convert_json_to_coefficients(j["coefficients_minus2"]);
}

// Function to load data from JSON files
void load_data_2_layer(
    RCP<const Basic>& symbolic_sum_no_mean,
    int& x_data_size,
    RCP<const Basic>& aux_all_sub_expressions_equation,
    unordered_map<RCP<const Basic>, RCP<const Basic>>& aux_dict_final,
    vector<vector<RCP<const Basic>>>& coefficients_plus1,
    vector<vector<RCP<const Basic>>>& coefficients_plus2,
    vector<vector<RCP<const Basic>>>& coefficients_plus3,
    const string& filename
) {
    json j = load_json_from_file(filename);

    symbolic_sum_no_mean = SymEngine::parse(j["symbolic_sum_no_mean"].get<string>());
    x_data_size = j["x_data_size"].get<int>();
    aux_all_sub_expressions_equation = SymEngine::parse(j["aux_all_sub_expressions_equation"].get<string>());

    for (const auto& pair : j["aux_dict_final"].items()) {
        aux_dict_final[SymEngine::parse(pair.key())] = SymEngine::parse(pair.value().get<string>());
    }

    auto convert_json_to_coefficients = [](const json& coeffs_json) {
        vector<vector<RCP<const Basic>>> coefficients;
        for (const auto& row : coeffs_json.items()) {
            vector<RCP<const Basic>> coeff_row;
            for (const auto& coeff : row.value().items()) {
                coeff_row.push_back(SymEngine::parse(coeff.value().get<string>()));
            }
            coefficients.push_back(coeff_row);
        }
        return coefficients;
    };

    coefficients_plus1 = convert_json_to_coefficients(j["coefficients_plus1"]);
    coefficients_plus2 = convert_json_to_coefficients(j["coefficients_plus2"]);
    coefficients_plus3 = convert_json_to_coefficients(j["coefficients_plus3"]);
}

// // Function to precompute powers
// unordered_map<int, RCP<const Basic>> precompute_powers(const RCP<const Basic>& expr, int max_power) {
//     unordered_map<int, RCP<const Basic>> powers;
//     for (int i = 1; i <= max_power; ++i) {
//         powers[i] = replace_binary_powers(expand(pow(expr, integer(i))));
//     }
//     return powers;
// }

// Function to precompute powers and apply auxiliary variables
unordered_map<int, RCP<const Basic>> precompute_powers(const RCP<const Basic>& expr, int max_power, unordered_map<RCP<const Basic>, RCP<const Basic>>& aux_dict, bool isTop) {
    unordered_map<int, RCP<const Basic>> powers;
    for (int i = 1; i <= max_power; ++i) {
        auto expanded_expr = replace_binary_powers(expand(pow(expr, integer(i))));
        auto result = apply_aux_variables(expanded_expr, aux_dict, isTop);
        powers[i] = result.first;
        aux_dict = result.second;
    }
    return powers;
}


RCP<const Basic> generate_coefficient_expr(const vector<vector<RCP<const Basic>>>& coefficients_plus, int degree, int m_current, int index) {
    RCP<const Basic> coeff = zero;
    for (int j = -m_current + 1; j <= 0; ++j) {
        int adj_index = j + m_current - 1; // Adjust the index to access the vectors correctly
        coeff = add(coeff, mul(pow(rational(2, 1), integer(j)), coefficients_plus[index][adj_index]));
    }
    return coeff;
}

RCP<const Basic> generate_coefficient_expr_rad2(const vector<vector<RCP<const Basic>>>& coefficients_plus, const vector<vector<RCP<const Basic>>>& coefficients_minus, int degree, int m_current, int index) {
    RCP<const Basic> coeff = zero;
    for (int j = -m_current; j <= m_current; ++j) {
        int idx = j + m_current; // Adjust the index to access the vectors correctly
        RCP<const Basic> term_plus = mul(pow(integer(2), integer(j)), coefficients_plus[index][idx]);
        RCP<const Basic> term_minus = mul(pow(integer(2), integer(j)), coefficients_minus[index][idx]);
        coeff = add(coeff, sub(term_plus, term_minus));
    }
    return coeff;
}

RCP<const Basic> substitute_precomputed_powers(const RCP<const Basic>& expr, const unordered_map<string, unordered_map<int, RCP<const Basic>>>& precomputed_powers) {
    if (is_a<Pow>(*expr)) {
        const auto& pow_expr = down_cast<const Pow&>(*expr);
        const auto& base = pow_expr.get_base();
        const auto& exp = pow_expr.get_exp();
        
        if (is_a<Symbol>(*base) && is_a<Integer>(*exp)) {
            const auto& base_str = down_cast<const Symbol&>(*base).get_name();
            const int exp_int = rcp_static_cast<const Integer>(exp)->as_int();
            
            if (precomputed_powers.count(base_str) && precomputed_powers.at(base_str).count(exp_int)) {
                return precomputed_powers.at(base_str).at(exp_int);
            }
        }
    }

    // Recursively apply substitution to arguments
    map_basic_basic new_subs;
    for (const auto& arg : expr->get_args()) {
        new_subs[arg] = substitute_precomputed_powers(arg, precomputed_powers);
    }
    return expr->subs(new_subs);
}

std::tuple<std::string, std::unordered_map<std::string, std::string>, std::vector<std::vector<std::string>>, std::vector<std::vector<std::string>>, std::vector<std::vector<std::string>>, std::vector<std::vector<std::string>>>
compute_mse_with_penalty_categorical(int d1, int d2, int m1, int m2, double penalty_multiplier, double bias_coefficient, const std::vector<double>& x_data_train, const std::vector<double>& y_data_train, const std::vector<double>& z_data_train, const std::vector<double>& x_data_test, const std::vector<double>& y_data_test, const std::vector<double>& z_data_test, double test_multiplier,  const std::string& load_filename = "", const std::string& save_filename = "") {
    int degree1 = d1;
    int degree2 = d2;

    auto start_total = std::chrono::high_resolution_clock::now(); // End total timing
    auto start_func_build = std::chrono::high_resolution_clock::now(); // End timing extraction

    // If a load_filename is provided, load the state from the file
    RCP<const Basic> aux_all_sub_expressions_equation;
    RCP<const Basic> symbolic_sum;
    RCP<const Basic> preloaded_symbolic_sum;
    unordered_map<RCP<const Basic>, RCP<const Basic>> aux_dict_final;
    vector<vector<RCP<const Basic>>> coefficients_plus1, coefficients_minus1, coefficients_plus2, coefficients_minus2;

    if (!load_filename.empty()) {
        load_data(symbolic_sum, aux_all_sub_expressions_equation, aux_dict_final, coefficients_plus1, coefficients_minus1, coefficients_plus2, coefficients_minus2, load_filename);
        preloaded_symbolic_sum = symbolic_sum;
    } else {
        // Define symbolic binary variables for the coefficients
        coefficients_plus1.resize(degree1 + 1, vector<RCP<const Basic>>(((2 * m1) + 1)));
        coefficients_minus1.resize(degree1 + 1, vector<RCP<const Basic>>(((2 * m1) + 1)));
        coefficients_plus2.resize(degree2 + 1, vector<RCP<const Basic>>(((2 * m2) + 1)));
        coefficients_minus2.resize(degree2 + 1, vector<RCP<const Basic>>(((2 * m2) + 1)));

        for (int i = 0; i <= degree1; ++i) {
            for (int j = 0; j < ((2 * m1) + 1); ++j) {
                coefficients_plus1[i][j] = binary("P1_" + to_string(i) + "_plus_" + to_string(j));
                coefficients_minus1[i][j] = binary("P1_" + to_string(i) + "_minus_" + to_string(j));
            }
        }

        for (int i = 0; i <= degree2; ++i) {
            for (int j = 0; j < ((2 * m2) + 1); ++j) {
                coefficients_plus2[i][j] = binary("P2_" + to_string(i) + "_plus_" + to_string(j));
                coefficients_minus2[i][j] = binary("P2_" + to_string(i) + "_minus_" + to_string(j));
            }
        }

        // Define control points
        vector<RCP<const Basic>> coefficients_A;
        vector<RCP<const Basic>> coefficients_B;

        for (int i = 0; i <= degree1; ++i) {
            coefficients_A.push_back(symbol("A" + std::to_string(i)));
        }

        for (int i = 0; i <= degree2; ++i) {
            coefficients_B.push_back(symbol("B" + std::to_string(i)));
        }

        // Generate coefficient expressions
        vector<RCP<const Basic>> coeff_expressions1(degree1 + 1);
        vector<RCP<const Basic>> coeff_expressions2(degree2 + 1);

        for (int i = 0; i <= degree1; ++i) {
            coeff_expressions1[i] = generate_coefficient_expr_rad2(coefficients_plus1, coefficients_minus1, degree1, m1, i);
        }

        for (int i = 0; i <= degree2; ++i) {
            coeff_expressions2[i] = generate_coefficient_expr_rad2(coefficients_plus2, coefficients_minus2, degree2, m2, i);
        }

        // Create an empty aux_dict
        unordered_map<RCP<const Basic>, RCP<const Basic>> existing_aux_dict_precomputed_powers;

        // Precompute powers
        unordered_map<string, unordered_map<int, RCP<const Basic>>> precomputed_powers;
        for (int i = 0; i <= degree1; ++i) {
            precomputed_powers["A" + to_string(i)] = precompute_powers(coeff_expressions1[i], degree1, existing_aux_dict_precomputed_powers, false);
        }
        for (int i = 0; i <= degree2; ++i) {
            precomputed_powers["B" + to_string(i)] = precompute_powers(coeff_expressions2[i], degree2, existing_aux_dict_precomputed_powers, false);
        }


        // Define symbolic variables
        RCP<const Basic> x = symbol("x");
        RCP<const Basic> y = symbol("y");
        RCP<const Basic> z = symbol("z");

        auto continuous_bezier_expr1 = bernstein_basis_functions_symbolic_continuous_control(x, degree1, coefficients_A);
        auto continuous_bezier_expr2 = bernstein_basis_functions_symbolic_continuous_control(y, degree2, coefficients_B);

        // Combine the two Bézier functions
        auto combined_expr = expand(add(continuous_bezier_expr1, continuous_bezier_expr2));

        // Do more powers
        auto expr_layer1_2 = expand(pow(combined_expr, integer(2)));

        // Substitute precomputed powers in substituted_expr
        expr_layer1_2 = substitute_precomputed_powers(expr_layer1_2, precomputed_powers);


        map_basic_basic substitutions_A;
        // Substitute coefficients in substituted_expr
        for (int i = 0; i <= degree1; ++i) {
            substitutions_A[symbol("A" + to_string(i))] = coeff_expressions1[i];
        }

        map_basic_basic substitutions_B;
        for (int i = 0; i <= degree2; ++i) {
            substitutions_B[symbol("B" + to_string(i))] = coeff_expressions2[i];
        }

        expr_layer1_2 = expr_layer1_2->subs(substitutions_A);
        expr_layer1_2 = expand(expr_layer1_2->subs(substitutions_B));

        // Now define the z^2
        auto z_squared= pow(z, integer(2));

        // Now define the middle expression
        auto middle_expression = expand(mul(mul(z, integer(-2)), combined_expr));
        
        middle_expression = middle_expression->subs(substitutions_A);

        middle_expression = expand(middle_expression->subs(substitutions_B));

        // Putting it all together:
        aux_all_sub_expressions_equation = add(add(z_squared, middle_expression), expr_layer1_2);

        // cout >> "aux_all_sub_expressions_equation: " << *aux_all_sub_expressions_equation << endl;
    }

    auto end_func_build = std::chrono::high_resolution_clock::now(); // End timing extraction

    auto start_unique_terms = std::chrono::high_resolution_clock::now(); // Start timing precompute

    auto unique_terms = extract_unique_xyz_terms(aux_all_sub_expressions_equation);

    auto end_unique_terms = std::chrono::high_resolution_clock::now(); // End timing extraction

    auto start_decompose = std::chrono::high_resolution_clock::now(); // Start timing precompute

    // Decompose the main expression into sub-expressions
    auto sub_expressions = separate_sub_expressions(aux_all_sub_expressions_equation);

    auto end_decompose = std::chrono::high_resolution_clock::now(); // Start timing precompute

    auto start_mapping = std::chrono::high_resolution_clock::now(); // Start timing precompute

    // Map xyz expressions to P_var expressions
    auto xyz_to_pvars = map_xyz_to_pvars(sub_expressions);

    auto end_mapping = std::chrono::high_resolution_clock::now(); // Start timing precompute

    auto start_precompute = std::chrono::high_resolution_clock::now(); // Start timing precompute
    auto start_eval = std::chrono::high_resolution_clock::now(); // Start timing precompute

    // Convert input vectors to Eigen arrays with memory alignment
    Eigen::ArrayXd x_train_eigen = Eigen::Map<const Eigen::ArrayXd, Eigen::Aligned>(x_data_train.data(), x_data_train.size());
    Eigen::ArrayXd y_train_eigen = Eigen::Map<const Eigen::ArrayXd, Eigen::Aligned>(y_data_train.data(), y_data_train.size());
    Eigen::ArrayXd z_train_eigen = Eigen::Map<const Eigen::ArrayXd, Eigen::Aligned>(z_data_train.data(), z_data_train.size());

    // Evaluate expressions for training data
    unordered_map<RCP<const Basic>, Eigen::ArrayXd, BasicHash, BasicEqual> evaluated_xyz_expressions_train;
    int max_degree = max(d1, d2); // Use std::max to get the maximum value
    int max_exp = max_degree * 2;

    evaluate_unique_xyz_expressions_optimized(xyz_to_pvars, x_train_eigen, y_train_eigen, z_train_eigen, evaluated_xyz_expressions_train, max_exp);
    RCP<const Basic> symbolic_sum_train = evaluate_and_combine(xyz_to_pvars, evaluated_xyz_expressions_train);

    Eigen::ArrayXd x_test_eigen = Eigen::Map<const Eigen::ArrayXd, Eigen::Aligned>(x_data_test.data(), x_data_test.size());
    Eigen::ArrayXd y_test_eigen = Eigen::Map<const Eigen::ArrayXd, Eigen::Aligned>(y_data_test.data(), y_data_test.size());
    Eigen::ArrayXd z_test_eigen = Eigen::Map<const Eigen::ArrayXd, Eigen::Aligned>(z_data_test.data(), z_data_test.size());

    // Evaluate expressions for test data
    unordered_map<RCP<const Basic>, Eigen::ArrayXd, BasicHash, BasicEqual> evaluated_xyz_expressions_test;

    evaluate_unique_xyz_expressions_optimized(xyz_to_pvars, x_test_eigen, y_test_eigen, z_test_eigen, evaluated_xyz_expressions_test, max_exp);

    auto end_eval = std::chrono::high_resolution_clock::now(); // Start timing precompute
    auto start_combineeval = std::chrono::high_resolution_clock::now(); // Start timing precompute

    RCP<const Basic> symbolic_sum_test = evaluate_and_combine(xyz_to_pvars, evaluated_xyz_expressions_test);

    // calculate the number of samples so SSE is MSE
    double mean_transformer = 1.0 / x_data_test.size();

    test_multiplier = test_multiplier * mean_transformer;
    // Apply the test multiplier to the test symbolic sum
    symbolic_sum_test = expand(mul(real_double(test_multiplier), symbolic_sum_test));
    symbolic_sum_train = expand(mul(real_double(mean_transformer), symbolic_sum_train));
    
    // Combine the training and test symbolic sums
    symbolic_sum = add(symbolic_sum_train, symbolic_sum_test);
    // symbolic_sum = symbolic_sum_train;
    auto end_combineeval = std::chrono::high_resolution_clock::now(); // Start timing precompute

    if (!load_filename.empty()) {
        cout << "combining the old symbolic_sum with the new one" << endl;
        symbolic_sum = add(symbolic_sum, preloaded_symbolic_sum);
    }

    // Find the largest coefficient
    double max_coeff = find_max_coefficient(symbolic_sum);

    auto start_penalty = std::chrono::high_resolution_clock::now(); // Start timing precompute

    // A good initial guess for a penalty coeff is 10x that of the largest coeff in the sse
    double penalty_coefficient = penalty_multiplier * max_coeff;

    // Generate penalty functions
    auto penalty_functions = generate_penalty_functions(aux_dict_final, penalty_coefficient);

    RCP<const Basic> sse_with_penalty = symbolic_sum;
    for (const auto& penalty_function : penalty_functions) {
        sse_with_penalty = add(sse_with_penalty, penalty_function);
    }

    auto end_penalty = std::chrono::high_resolution_clock::now(); // Start timing precompute

    auto end_precompute = std::chrono::high_resolution_clock::now(); // End timing precompute

    auto start_str_substitution = std::chrono::high_resolution_clock::now(); // Start timing substitution

    // Convert sse_with_penalty to string
    std::string sse_with_penalty_str = sse_with_penalty->__str__();

    // Convert aux_dict_final to a map of strings for easier handling in Python
    std::unordered_map<std::string, std::string> aux_dict_str;
    for (const auto& pair : aux_dict_final) {
        aux_dict_str[pair.first->__str__()] = pair.second->__str__();
    }

    // Convert coefficients to strings
    auto convert_coefficients_to_strings = [](const std::vector<std::vector<RCP<const Basic>>>& coefficients) {
        std::vector<std::vector<std::string>> coeffs_str;
        for (const auto& row : coefficients) {
            std::vector<std::string> row_str;
            for (const auto& coeff : row) {
                row_str.push_back(coeff->__str__());
            }
            coeffs_str.push_back(row_str);
        }
        return coeffs_str;
    };

    std::vector<std::vector<std::string>> coeffs_plus1_str = convert_coefficients_to_strings(coefficients_plus1);
    std::vector<std::vector<std::string>> coeffs_plus2_str = convert_coefficients_to_strings(coefficients_plus2);
    std::vector<std::vector<std::string>> coeffs_minus1_str = convert_coefficients_to_strings(coefficients_minus1);
    std::vector<std::vector<std::string>> coeffs_minus2_str = convert_coefficients_to_strings(coefficients_minus2);

    // Save the current state if a save_filename is provided
    if (!save_filename.empty()) {
        save_data(symbolic_sum, aux_all_sub_expressions_equation, aux_dict_final, coefficients_plus1, coefficients_minus1, coefficients_plus2, coefficients_minus2, save_filename);
    }
    auto end_str_substitution = std::chrono::high_resolution_clock::now(); // End timing substitution
    auto end_total = std::chrono::high_resolution_clock::now(); // End total timing

    // Calculate and print elapsed times
    std::chrono::duration<double> elapsed_func_build = end_func_build - start_func_build;
    std::chrono::duration<double> elapsed_precompute = end_precompute - start_precompute;
    std::chrono::duration<double> elapsed_unique_terms = end_unique_terms - start_unique_terms;
    std::chrono::duration<double> elapsed_str_substitution = end_str_substitution - start_str_substitution;
    std::chrono::duration<double> elapsed_decomposed = end_decompose - start_decompose;
    std::chrono::duration<double> elapsed_mapping = end_mapping - start_mapping;
    std::chrono::duration<double> elapsed_eval = end_eval - start_eval;
    std::chrono::duration<double> elapsed_combineeval = end_combineeval - start_combineeval;
    std::chrono::duration<double> elapsed_penalty = end_penalty - start_penalty;
    std::chrono::duration<double> elapsed_total = end_total - start_total;

    cout << "Time taken for function building: " << elapsed_func_build.count() << " seconds" << endl;
    cout << "Time taken for elapsed_unique_terms: " << elapsed_unique_terms.count() << " seconds" << endl;
    cout << "Time taken for elapsed_decomposed: " << elapsed_decomposed.count() << " seconds" << endl;
    cout << "Time taken for elapsed_mapping: " << elapsed_mapping.count() << " seconds" << endl;
    cout << "Time taken for elapsed_eval: " << elapsed_eval.count() << " seconds" << endl;
    cout << "Time taken for elapsed_combineeval: " << elapsed_combineeval.count() << " seconds" << endl;
    cout << "Time taken for elapsed_penalty: " << elapsed_penalty.count() << " seconds" << endl;
    cout << "Time taken for precomputation of values: " << elapsed_precompute.count() << " seconds" << endl;
    cout << "Time taken for string substitution: " << elapsed_str_substitution.count() << " seconds" << endl;
    cout << "Total time taken: " << elapsed_total.count() << " seconds" << endl;

    return std::make_tuple(sse_with_penalty_str, aux_dict_str, coeffs_plus1_str, coeffs_minus1_str, coeffs_plus2_str, coeffs_minus2_str);
}

std::tuple<std::string, std::unordered_map<std::string, std::string>, std::vector<std::vector<std::string>>, std::vector<std::vector<std::string>>, std::vector<std::vector<std::string>>>
compute_mse_with_penalty(int d1, int d2, int d3, int m1, int m2, int m3, double penalty_multiplier, double bias_coefficient, bool is_fractional, const std::vector<double>& x_data, const std::vector<double>& y_data, const std::vector<double>& z_data,  const std::string& load_filename = "", const std::string& save_filename = "") {
    int degree1 = d1;
    int degree2 = d2;
    int degree3 = d3;

    // If a load_filename is provided, load the state from the file
    RCP<const Basic> aux_all_sub_expressions_equation;
    RCP<const Basic> symbolic_sum_no_mean;
    RCP<const Basic> preloaded_symbolic_sum;
    unordered_map<RCP<const Basic>, RCP<const Basic>> aux_dict_final;
    vector<vector<RCP<const Basic>>> coefficients_plus1, coefficients_plus2, coefficients_plus3;
    int x_data_size = x_data.size();
    int x_data_size_old;

    if (!load_filename.empty()) {
        load_data_2_layer(preloaded_symbolic_sum, x_data_size_old, aux_all_sub_expressions_equation, aux_dict_final, coefficients_plus1, coefficients_plus2, coefficients_plus3, load_filename);
        x_data_size = x_data_size + x_data_size_old;
    } else {
        // Define symbolic binary variables for the coefficients
        coefficients_plus1.resize(degree1 + 1, vector<RCP<const Basic>>(m1));
        coefficients_plus2.resize(degree2 + 1, vector<RCP<const Basic>>(m2));
        coefficients_plus3.resize(degree3 + 1, vector<RCP<const Basic>>(m3));
        // initialize_coefficients();

        for (int i = 0; i <= degree1; ++i) {
            for (int j = 0; j < m1; ++j) {
                coefficients_plus1[i][j] = binary("P1_" + to_string(i) + "_plus_" + to_string(j));
            }
        }

        for (int i = 0; i <= degree2; ++i) {
            for (int j = 0; j < m2; ++j) {
                coefficients_plus2[i][j] = binary("P2_" + to_string(i) + "_plus_" + to_string(j));
            }
        }

        for (int i = 0; i <= degree3; ++i) {
            for (int j = 0; j < m3; ++j) {
                coefficients_plus3[i][j] = binary("P3_" + to_string(i) + "_plus_" + to_string(j));
            }
        }
        // Define control points
        vector<RCP<const Basic>> coefficients_A;
        vector<RCP<const Basic>> coefficients_B;
        vector<RCP<const Basic>> coefficients_C;

        for (int i = 0; i <= degree1; ++i) {
            coefficients_A.push_back(symbol("A" + std::to_string(i)));
        }

        for (int i = 0; i <= degree2; ++i) {
            coefficients_B.push_back(symbol("B" + std::to_string(i)));
        }

        for (int i = 0; i <= degree3; ++i) {
            coefficients_C.push_back(symbol("C" + std::to_string(i)));
        }

        // Generate coefficient expressions
        vector<RCP<const Basic>> coeff_expressions1(degree1 + 1);
        vector<RCP<const Basic>> coeff_expressions2(degree2 + 1);
        vector<RCP<const Basic>> coeff_expressions3(degree3 + 1);

        for (int i = 0; i <= degree1; ++i) {
            coeff_expressions1[i] = generate_coefficient_expr(coefficients_plus1, degree1, m1, i);
        }

        for (int i = 0; i <= degree2; ++i) {
            coeff_expressions2[i] = generate_coefficient_expr(coefficients_plus2, degree2, m2, i);
        }

        for (int i = 0; i <= degree3; ++i) {
            coeff_expressions3[i] = generate_coefficient_expr(coefficients_plus3, degree3, m3, i);
        }

        // Create an empty aux_dict
        unordered_map<RCP<const Basic>, RCP<const Basic>> existing_aux_dict_precomputed_powers;

        // Precompute powers
        unordered_map<string, unordered_map<int, RCP<const Basic>>> precomputed_powers;
        for (int i = 0; i <= degree1; ++i) {
            precomputed_powers["A" + to_string(i)] = precompute_powers(coeff_expressions1[i], 8, existing_aux_dict_precomputed_powers, true);
        }
        for (int i = 0; i <= degree2; ++i) {
            precomputed_powers["B" + to_string(i)] = precompute_powers(coeff_expressions2[i], 8, existing_aux_dict_precomputed_powers, true);
        }
        for (int i = 0; i <= degree3; ++i) {
            precomputed_powers["C" + to_string(i)] = precompute_powers(coeff_expressions3[i], 8, existing_aux_dict_precomputed_powers, true);
        }

        // Define symbolic variables
        RCP<const Basic> x = symbol("x");
        RCP<const Basic> y = symbol("y");
        RCP<const Basic> t = symbol("t");
        RCP<const Basic> z = symbol("z");

        // Compute the symbolic basis functions
        auto continuous_bezier_expr1 = bernstein_basis_functions_symbolic_continuous_control(x, degree1, coefficients_A);
        auto continuous_bezier_expr2 = bernstein_basis_functions_symbolic_continuous_control(y, degree2, coefficients_B);
        auto continuous_bezier_expr3 = bernstein_basis_functions_symbolic_continuous_control(t, degree3, coefficients_C);

        RCP<const Basic> bezier_expr1, bezier_expr2, bezier_expr3;

        // Combine the two Bézier functions
        auto combined_continuous_bottom_expr = expand(add(continuous_bezier_expr1, continuous_bezier_expr2));

        // Create the power of the third Bézier function
        auto bezier_continuous_expr3_2 = expand(pow(continuous_bezier_expr3, integer(2)));

        // Substitute combined_continuous_bottom_expr for t in bezier_continuous_expr3_2
        map_basic_basic substitutions;
        substitutions[t] = combined_continuous_bottom_expr;
        auto substituted_expr = expand(bezier_continuous_expr3_2->subs(substitutions));

        // Substitute precomputed powers in substituted_expr
        auto final_expr = substitute_precomputed_powers(substituted_expr, precomputed_powers);

        map_basic_basic substitutions_A;

        // Substitute coefficients in substituted_expr
        for (int i = 0; i <= degree1; ++i) {
            substitutions_A[symbol("A" + to_string(i))] = coeff_expressions1[i];
        }

        substituted_expr = final_expr->subs(substitutions_A);

        map_basic_basic substitutions_C;

        for (int i = 0; i <= degree3; ++i) {
            substitutions_C[symbol("C" + to_string(i))] = coeff_expressions3[i];
        }
        substituted_expr = substituted_expr->subs(substitutions_C);

        map_basic_basic substitutions_B;

        for (int i = 0; i <= degree2; ++i) {
            substitutions_B[symbol("B" + to_string(i))] = coeff_expressions2[i];
        }

        auto final_substituted_expr = expand(substituted_expr->subs(substitutions_B));

        // Create auxiliary variables
        auto result = apply_aux_variables(final_substituted_expr, existing_aux_dict_precomputed_powers, false);
        final_substituted_expr = result.first;
        existing_aux_dict_precomputed_powers = result.second;

        // Now define the z^2
        auto z_squared= pow(z, integer(2));

        // Now define the middle expression
        auto middle_expression = mul(mul(z, integer(-2)), continuous_bezier_expr3);

        map_basic_basic substitutions_middle;
        substitutions_middle[t] = combined_continuous_bottom_expr;
        auto substituted_expr_middle = expand(middle_expression->subs(substitutions_middle));

        // Substitute precomputed powers in substituted_expr
        auto final_expr_middle = substitute_precomputed_powers(substituted_expr_middle, precomputed_powers);

        final_expr_middle = final_expr_middle->subs(substitutions_A);

        final_expr_middle = final_expr_middle->subs(substitutions_C);

        final_expr_middle = expand(final_expr_middle->subs(substitutions_B));

        // Create auxiliary variables
        result = apply_aux_variables(final_expr_middle, existing_aux_dict_precomputed_powers, false);
        final_expr_middle = result.first;
        aux_dict_final = result.second;

        // Putting it all together:
        aux_all_sub_expressions_equation = add(add(z_squared, final_expr_middle), final_substituted_expr);

        // Filter the auxiliary dictionary
        filter_aux_dict(aux_all_sub_expressions_equation, aux_dict_final);
    }


    auto unique_terms = extract_unique_xyz_terms(aux_all_sub_expressions_equation);


    // Decompose the main expression into sub-expressions
    auto unique_sub_expressions = separate_sub_expressions(aux_all_sub_expressions_equation);

    // Map xyz expressions to P_var expressions
    auto xyz_to_pvars = map_xyz_to_pvars(unique_sub_expressions);

    // Precompute unique xyz expressions
    unordered_map<RCP<const Basic>, Eigen::ArrayXd, BasicHash, BasicEqual> evaluated_xyz_expressions;

    // Convert input vectors to Eigen arrays with memory alignment
    Eigen::ArrayXd x_eigen = Eigen::Map<const Eigen::ArrayXd, Eigen::Aligned>(x_data.data(), x_data.size());
    Eigen::ArrayXd y_eigen = Eigen::Map<const Eigen::ArrayXd, Eigen::Aligned>(y_data.data(), y_data.size());
    Eigen::ArrayXd z_eigen = Eigen::Map<const Eigen::ArrayXd, Eigen::Aligned>(z_data.data(), z_data.size());

    // Precompute unique xyz expressions
    int max_degree = max(d1, d2); // Use std::max to get the maximum value
    int max_exp = max_degree * d3 * 2;

    evaluate_unique_xyz_expressions_optimized(xyz_to_pvars, x_eigen, y_eigen, z_eigen, evaluated_xyz_expressions, max_exp);
    
    // Evaluate and combine the expressions
    symbolic_sum_no_mean = evaluate_and_combine(xyz_to_pvars, evaluated_xyz_expressions);

    // calculate the number of samples so SSE is MSE
    double mean_transformer = 1.0 / x_data_size;

    if (!load_filename.empty()) {
        symbolic_sum_no_mean = add(symbolic_sum_no_mean, preloaded_symbolic_sum);
    }

    // Apply the test multiplier to the test symbolic sum
    RCP<const Basic> symbolic_sum = expand(mul(real_double(mean_transformer), symbolic_sum_no_mean));

    // Find the largest coefficient
    double max_coeff = find_max_coefficient(symbolic_sum);

    // A good initial guess for a penalty coeff is 10x that of the largest coeff in the sse
    double penalty_coefficient = penalty_multiplier * max_coeff;

    // Generate penalty functions
    auto penalty_functions = generate_penalty_functions(aux_dict_final, penalty_coefficient);

    RCP<const Basic> sse_with_penalty = symbolic_sum;
    for (const auto& penalty_function : penalty_functions) {
        sse_with_penalty = add(sse_with_penalty, penalty_function);
    }

    // Convert sse_with_penalty to string
    std::string sse_with_penalty_str = sse_with_penalty->__str__();

    // Convert aux_dict_final to a map of strings for easier handling in Python
    std::unordered_map<std::string, std::string> aux_dict_str;
    for (const auto& pair : aux_dict_final) {
        aux_dict_str[pair.first->__str__()] = pair.second->__str__();
    }

    // Convert coefficients to strings
    auto convert_coefficients_to_strings = [](const std::vector<std::vector<RCP<const Basic>>>& coefficients) {
        std::vector<std::vector<std::string>> coeffs_str;
        for (const auto& row : coefficients) {
            std::vector<std::string> row_str;
            for (const auto& coeff : row) {
                row_str.push_back(coeff->__str__());
            }
            coeffs_str.push_back(row_str);
        }
        return coeffs_str;
    };

    std::vector<std::vector<std::string>> coeffs_plus1_str = convert_coefficients_to_strings(coefficients_plus1);
    std::vector<std::vector<std::string>> coeffs_plus2_str = convert_coefficients_to_strings(coefficients_plus2);
    std::vector<std::vector<std::string>> coeffs_plus3_str = convert_coefficients_to_strings(coefficients_plus3);

    // Save the current state if a save_filename is provided
    if (!save_filename.empty()) {
        save_data_2_layer(symbolic_sum_no_mean, x_data_size, aux_all_sub_expressions_equation, aux_dict_final, coefficients_plus1, coefficients_plus2, coefficients_plus3, save_filename);
    }

    return std::make_tuple(sse_with_penalty_str, aux_dict_str, coeffs_plus1_str, coeffs_plus2_str, coeffs_plus3_str);
}

PYBIND11_MODULE(quantum_kan, m) {
    m.def("compute_mse_with_penalty_categorical", &compute_mse_with_penalty_categorical, "Compute MSE with penalty for categorical",
          py::arg("d1"), py::arg("d2"),
          py::arg("m1"), py::arg("m2"),
          py::arg("penalty_multiplier"),
          py::arg("bias_coefficient"),
          py::arg("x_data_train"), py::arg("y_data_train"), py::arg("z_data_train"),
          py::arg("x_data_test"), py::arg("y_data_test"), py::arg("z_data_test"),
          py::arg("test_multiplier"),
          py::arg("load_filename") = "",
          py::arg("save_filename") = "");

    m.def("compute_mse_with_penalty", &compute_mse_with_penalty, "Compute MSE with penalty",
          py::arg("d1"), py::arg("d2"), py::arg("d3"),
          py::arg("m1"), py::arg("m2"), py::arg("m3"),
          py::arg("penalty_multiplier"),
          py::arg("bias_coefficient"),
          py::arg("is_fractional"),
          py::arg("x_data"), py::arg("y_data"), py::arg("z_data"),
          py::arg("load_filename") = "",
          py::arg("save_filename") = "");
}
