/* stdfin random/non_central_chi_squared_distribution.hpp header file
 *
 * Copyright Thijs van den Berg 2014-2015
 * 
 * Distributed under the MIT Software License.
 * See the accompanying file LICENSE or copy at http://opensource.org/licenses/MIT
 *
 */

#ifndef STDFIN_RANDOM_NON_CENTRAL_CHI_SQUARED_DISTRIBUTION_HPP
#define STDFIN_RANDOM_NON_CENTRAL_CHI_SQUARED_DISTRIBUTION_HPP

#include <iosfwd>
#include <cmath>
#include <random>

namespace stdfin {

/**
 * The noncentral chi-squared distribution is a real valued distribution with
 * two parameter, @c k and @c lambda.  The distribution produces values > 0.
 *
 * This is the distribution of the sum of squares of k Normal distributed variates each with variance one 
 * and \f$\lambda\f$ the sum of squares of the normal means.
 *
 * The distribution function is
 * \f$\displaystyle P(x) = \frac{1}{2} e^{-(x+\lambda)/2} \left( \frac{x}{\lambda} \right)^{k/4-1/2} I_{k/2-1}( \sqrt{\lambda x} )\f$.
 *  where  \f$\displaystyle I_\nu(z)\f$ is a modified Bessel function of the first kind.
 */
template <typename RealType = double>
class non_central_chi_squared_distribution {
public:
    typedef RealType result_type;
    typedef RealType input_type;
    typedef non_central_chi_squared_distribution<RealType> distribution_type;
    
    class param_type {
    public:
        typedef non_central_chi_squared_distribution distribution_type;
        
        explicit
        param_type(RealType k_arg = RealType(1), RealType lambda_arg = RealType(1))
        : _M_k(k_arg), _M_lambda(lambda_arg)
        { }
        
        /** Returns the @c k parameter of the distribution */
        RealType k() const { return _M_k; }
        
        /** Returns the @c lambda parameter of the distribution */
        RealType lambda() const { return _M_lambda; }
        
        /** Writes the parameters of the distribution to a @c std::ostream. */
        template<typename CharT, typename Traits> 
        friend std::basic_ostream< CharT, Traits >& 
        operator<<(std::basic_ostream< CharT, Traits > &os, const param_type& parm)
        {
            const typename std::ios_base::fmtflags saved_flags = os.flags();
            os.flags(std::ios_base::scientific);
            os << parm._M_k << ' ' << parm._M_lambda;
            os.flags(saved_flags);
            return os;
        }
        
        template<typename CharT, typename Traits> 
        friend std::basic_istream< CharT, Traits > & 
        operator>>(std::basic_istream< CharT, Traits > &is, param_type &parm)
        {
            const typename std::ios_base::fmtflags saved_flags = is.flags();
            is.flags(std::ios_base::scientific | std::ios_base::skipws);
            is >> parm._M_k >> parm._M_lambda;
            is.flags(saved_flags);
            return is;
        }

        /** Returns true if the parameters have the same values. */
        friend bool operator==(const param_type &lhs, const param_type &rhs)
        { return lhs._M_k == rhs._M_k && lhs._M_lambda == rhs._M_lambda; }
        
        /** Returns true if the parameters have different values. */
        friend bool operator!=(const param_type &lhs, const param_type &rhs)
        { return !(lhs == rhs); }
        
    private:
        RealType _M_k;
        RealType _M_lambda;
    };

    /**
     * Construct a @c non_central_chi_squared_distribution object. @c k and @ lambda
     * are the parameter of the distribution.
     *
     * Requires: (n > 0) && (lambda > 0)
     */
    explicit
    non_central_chi_squared_distribution(RealType k_arg = RealType(1), RealType lambda_arg = RealType(1))
    : _M_param(k_arg, lambda_arg)
    { }

    /**
     * Construct a @c non_central_chi_squared_distribution object from the parameter.
     */
    explicit
    non_central_chi_squared_distribution(const param_type& parm)
    : _M_param( parm )
    { }
    
    /**
     * Returns a random variate distributed according to the
     * non central chi squared distribution specified by @c param.
     */
    template<typename URNG>
    result_type operator()
    (URNG& eng, const param_type& parm) const
    { return non_central_chi_squared_distribution(parm)(eng); }
    
    /**
     * Returns a random variate distributed according to the
     * non central chi squared distribution.
     *
     * The algorithm is taken from Monte Carlo Methods in Financial Engineering, 
     * Paul Glasserman, p 124
     */
    template<typename URNG> 
    result_type operator()
    (URNG& eng) 
    {
        if (_M_param.k() > 1) {
            std::normal_distribution<RealType> n_dist;
            std::chi_squared_distribution<RealType> c_dist(_M_param.k() - RealType(1));
            RealType _z = n_dist(eng);
            RealType _x = c_dist(eng);
            RealType term1 = _z + std::sqrt(_M_param.lambda());
            return term1*term1 + _x;
        }
        else {
            std::poisson_distribution<> p_dist(_M_param.lambda()/RealType(2));
            std::poisson_distribution<>::result_type _p = p_dist(eng);
            std::chi_squared_distribution<RealType> c_dist(_M_param.k() + RealType(2)*_p);
            return c_dist(eng);
        }
    }

    /** Returns the @c k parameter of the distribution. */
    RealType k() const
    { return _M_param.k(); }
    
    /** Returns the @c lambda parameter of the distribution. */
    RealType lambda() const
    { return _M_param.lambda(); }
    
    /** Returns the parameters of the distribution. */
    param_type param() const
    { return _M_param; }
    
    /** Sets parameters of the distribution. */
    void param(const param_type& parm) 
    { _M_param = parm; }
    
    /** Resets the distribution, so that subsequent uses does not depend on values already produced by it.*/
    void reset() {}
    
    /** Returns the smallest value that the distribution can produce. */
    result_type (min)() const
    { return RealType(0); }
    
    /** Returns the largest value that the distribution can produce. */
    result_type (max)() const
    { return (std::numeric_limits<RealType>::infinity)(); }

    /** Writes the parameters of the distribution to a @c std::ostream. */
    template<typename CharT, typename Traits> 
    friend std::basic_ostream< CharT, Traits >& 
    operator<<(std::basic_ostream< CharT, Traits > &os, const non_central_chi_squared_distribution& dist)
    {
        const typename std::ios_base::fmtflags saved_flags = os.flags();
        os.flags(std::ios_base::scientific);
        os << dist.k() << ' ' << dist.lambda();
        os.flags(saved_flags);
        return os;
    }
    
    /** reads the parameters of the distribution from a @c std::istream. */
    template<typename CharT, typename Traits> 
    friend std::basic_istream< CharT, Traits > & 
    operator>>(std::basic_istream< CharT, Traits > &is, non_central_chi_squared_distribution &dist)
    {
        const typename std::ios_base::fmtflags saved_flags = is.flags();
        is.flags(std::ios_base::scientific | std::ios_base::skipws);
        RealType k_arg;
        RealType lambda_arg;
        is >> k_arg >> lambda_arg;
        dist.param( param_type(k_arg, lambda_arg) );
        is.flags(saved_flags);
        return is;
    }

    /** Returns true is two distributions have the same parameters and produce 
        the same sequence of random numbers given equal generators.*/
    friend bool operator==(const non_central_chi_squared_distribution &lhs, const non_central_chi_squared_distribution &rhs)
    { return lhs.param() == rhs.param(); }
    
    /** Returns true is two distributions have different parameters and/or can produce 
       different sequences of random numbers given equal generators.*/
    friend bool operator!=(const non_central_chi_squared_distribution &lhs, const non_central_chi_squared_distribution &rhs)
    { return !(lhs == rhs); }
    
private:

    /// @cond show_private
    param_type  _M_param;
    /// @endcond
};

typedef non_central_chi_squared_distribution<> non_central_chi_squared;

} // namespace stdfin

#endif
