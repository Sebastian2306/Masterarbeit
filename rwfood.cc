/*****************************************************************************/
/* simulation of a square-lattice random walker consuming food               */
/*****************************************************************************/
// this is C++11 code
// when using GCC, compile with g++ -std=c++11 -O3
// you can optionally use -fopenmp
#include <iostream>
#include <fstream>
#include <sstream>
#include <random>
#include <functional> // for std::bind and std::hash
// for production code, disable index-range checking etc. within boost
#define BOOST_DISABLE_ASSERTS
#include <boost/unordered_set.hpp>
#include <boost/multi_array.hpp>
#include <boost/range/iterator_range.hpp> // for boost::make_iterator_range
#ifdef BIAS_NUM_VISITS
#include <boost/unordered_map.hpp>
#endif

#ifdef TIMED_LOG
#include <chrono>
#endif
#include <stack>
#include <queue>
#include <iomanip>


/* code layout:

   (1) compile-time parameters (run lengths, dimension, etc.)
   (2) data storage typedefs
   (3) random number generator
   (4) auxiliary classes etc. (warning: heavy C++)
   (5) hashing of lattice sites visited by a random walk
   (6) global variables: lattice directions
   (7) global variables: results like MSD etc.
   (8) initialization (called before any of the runs)
   (9) food implementation
   (10) percolating matrix implementation
   (11) run(), the main code performing a single random walk
*/


/** (1) compile-time parameters **/

// Nruns: number of random-walk runs to perform per matrix realization
// Nmatrices: number of random matrices to generate
// length: number of steps of the single random walks
// dim: number of spatial dimensions
// rng_seed: random number generator seed, used once at program start
// calc_vanhove: boolean to control whether we evaluate the van Hove function

/* Those all have defaults that can be overruled by preprocessor defines
   (see the Makefile). Further things controlled by preprocessor defines:
   - The actual food implementation.
   - The calculation of van Hove functions (whether and which times).
   - The implementation of a percolating cluster.
   Rationale: Make those things change-able without changing this C++ source
   file.
   If we put this file under version control, it should not change its
   version only because we compile for different parameters!
   (Update, version 14: the food level and matrix-occupation probability
   are compile-time defaults that can be overruled at runtime.)
*/
#ifndef NRUNS
  #define NRUNS 1000
#endif
#ifndef LENGTH
  #define LENGTH 500000
#endif
#ifndef DIM
  #define DIM 1
#endif
#ifndef RNGSEED
  #define RNGSEED 20140901
#endif
#ifndef FOOD
  #define FOOD level_food(F)
#endif
#ifndef VANHOVE
  #define VANHOVE 0
#endif
#ifndef VANHOVE_TIMES
  #define VANHOVE_TIMES {}
#endif
#ifndef VANHOVE_CUTOFF
  #define VANHOVE_CUTOFF 500
#endif
#ifndef MATRIX
  #define MATRIX empty_space
#endif
#ifndef NMATRICES
  #define NMATRICES 1
#endif
#ifndef PROB
  #define PROB 0.6
#endif
#ifndef LX
  #define LX 20
#endif
#ifndef RANDOM_START
  #define RANDOM_START true
#endif
#ifndef RANDOM_START_CENTER
  #ifdef COUNT_BACKBONE
  #define RANDOM_START_CENTER true
  #else
  #define RANDOM_START_CENTER false
  #endif
#endif
#ifndef MAX_BUCKETS
  #define MAX_BUCKETS (1<<30)
#endif
#ifndef COUNT_BACKBONE
  #define COUNT_BACKBONE 0
#endif
// there is code activated by BACKBONE_EXP for a new algorithm
// history: the backbone code I use per default now was introduced in v48,
// and in v49 I thought to have a better variant, but discussing again
// with Thomas Franosch, I believe that the v48 code is still better
// this code was re-merged in v52 as the default

// actual C++ declarations of the compile-time parameters
static const size_t Nruns = NRUNS;
static const size_t Nmatrices = NMATRICES;
static const unsigned long length = LENGTH;
static const unsigned int dim = DIM;
uint32_t rng_seed = RNGSEED;
// control block-wise output: calculate/print observables in blocks of length B;
// the output step size is doubled after each such block
static const size_t B = 128;
//
static const bool calc_vanhove = VANHOVE;
static const size_t vanhove_cutoff = VANHOVE_CUTOFF;
//
static const bool count_backbone = COUNT_BACKBONE;
//
static std::array<unsigned long,dim> Lx = {LX};
static bool random_start = RANDOM_START;
static bool random_start_center = RANDOM_START_CENTER;

// runtime change-able parameters
static double p = PROB;
static double F = 1.0;

bool intermediate_averages = false;

// Allow runtime selection of parameters:
// admittedly, this is C inside C++.
// excuse: sscanf is rather powerful.
void parse_cmdline (int argc, char **argv) {
  char var [80], valbuf [256]; double val; int i;
  if (argc>1) for (i=1; i<argc; ++i) {
    if (sscanf(argv[i], "%40[a-zA-Z0-9] = %lf", (char*)(&var), &val)==2) {
      if (!strcmp(var,"p")) p = val;
      else if (!strcmp(var,"F")) F = val;
      else if (!strcmp(var,"printtmp")) intermediate_averages=val;
      else {
        std::cerr << "WARNING: ignored unknown variable '" << var << "'\n";
      }
    } else {
      // could implement reading a parameter file whose name would
      // be argv[i] at this point
      std::cerr << "WARNING: could not parse '" << argv[i] << "'" << std::endl;
    }
  }
}


/** (2) types for data storage **/

// pos_t: stores a single lattice position
// poslist_t: a list of positions (stores the random walk)
// pos_hash_t: a lookup table storing visited positions
typedef std::array<long,dim> pos_t;
typedef std::array<pos_t,length> poslist_t;
typedef boost::unordered_set<pos_t> pos_hash_t;
#ifdef BIAS_NUM_VISITS
typedef boost::unordered_map<pos_t,unsigned long> pos_count_t;
#endif

#include <cstdint>
template <int b> struct int_with_bits { typedef unsigned int type; };
template <> struct int_with_bits<32> { typedef uint32_t type; };
template <> struct int_with_bits<64> { typedef uint64_t type; };

/** (3) random number generation **/

// We use the standard Mersenne Twister as the PRNG,
// transformed into a uniform distribution on [0,1)
// see http://www.cplusplus.com/reference/random/
typedef std::mt19937 mersenne_twister;
std::uniform_real_distribution<double> uniform_distribution (0.0, 1.0);
mersenne_twister rng; // global rng object
// the following binds random_number() to do what we want...
// note the std::ref - otherwise, std::bind makes a copy of rng,
// and our later seeding will be without effect when calling random_number()
auto random_number = std::bind (uniform_distribution, std::ref(rng));


/** (4) auxiliary constants, classes, functions **/

/* The stuff here pulls a few C++ tricks. In the end, it just does what
   you'd intuitively expect (I hope), so don't worry... */

/* Here we collect some useful definitions that are not in the standard
   header files, including some operators allowing to print position vectors
   or memory sizes (just for convenience in later coding).
   We also add two expression-template classes that help in compile-time
   optimization: the first, _pow_, tells the compiler to decompose integer
   powers of the form (x^n), written as pow<n>(x), into an O(lg2(n)) number
   of multiplications. Think of pow<2>(x) as the C++-style way of implementing
   the common C-style macro #define SQR(x) ((x)*(x)).
   The second, _log2_ calculates floor(lg2(n)) of an integer n at compile time.
   Recall that for a walk of given length (compile-time parameter), we
   only need to store O((lg2(length/B)+1)*B) values for the MSD etc.; this
   value is calculated at compile time later on, using the template here.
   Admittedly, these expression templates are old-style C++ (C++03), while
   in C++11 they could be more compactly written using constexpr functions.
*/
// we might need the value of pi later on
static const double pi = 3.1415926535897932384626433832795028841971693993751;
// expression template to ask the compiler to evaluate
// x^n as x^[n/2]*x^[n/2]*x[n-n/2-n/2] ([n/2] denoting floor(n/2)) recursively.
template <unsigned int n, class T> struct _pow_ {
  static inline T apply (const T &val) {
    T tmp = _pow_<n/2,T>::apply(val);
    return tmp*tmp*_pow_<n-n/2-n/2,T>::apply(val);
  }
};
// terminate the recursion by setting x^1 = x and x^0 = 1
template <> template <class T> struct _pow_<1,T> {
  static inline T apply (const T &val) { return val; }
};
template <> template <class T> struct _pow_<0,T> {
  static inline T apply (const T &) { return 1; }
};
// in the end, say pow<2>(x) to get x^2 compiled as x*x
template <unsigned int n, class T>
inline T pow (const T &val) { return _pow_<n,T>::apply(val); }
// expression template to ask the compiler to evaluate floor(log2(val))
// this works by dividing by 2 repeatedly, counting the number of divisions
template <unsigned long val> struct _log2_ {
  constexpr static inline unsigned int apply () { return 1+_log2_<val/2>::apply(); }
};
template <> struct _log2_<2> {
  constexpr static inline unsigned int apply () { return 1; }
};
template <> struct _log2_<1> {
  constexpr static inline unsigned int apply () { return 0; }
};
// not mathematically correct, but catches the case where int(length/B)==0
template <> struct _log2_<0> {
  constexpr static inline unsigned int apply () { return 0; }
};
// implement shortcut for the addition of position vectors as x+y;
// the compiler should make this efficient inline and optimize away
// the temporary object called res
inline pos_t operator + (const pos_t &a, const pos_t &b) {
  pos_t res = a;
  for (unsigned int i=0; i<a.size(); ++i) res[i] += b[i];
  return res;
}
// implement printing of position vectors as cout << x,
// separating components by tabs
inline std::ostream &operator << (std::ostream &os, const pos_t &pos) {
  os << pos[0];
  if (pos.size()>1)
    for (unsigned int i=1; i<pos.size(); ++i) os << "\t" << pos[i];
  return os;
}
// implement pretty-printing of memory sizes
class memsize {
  static const char *units [];
  double msz;
  const char * const *u;
public:
  inline memsize (unsigned long count) : msz(count), u(units) {
    while (*(u+1) != 0 && msz>1024) { msz/=1024; ++u; }
  }
  inline std::ostream &print (std::ostream &os) const {
    os << msz << " " << *u;
    return os;
  }
};
const char *memsize::units [] = { "bytes", "kB", "MB", "GB", "TB", 0 };
inline std::ostream &operator << (std::ostream &os, const memsize &m) {
  return m.print(os);
}

/** (5) handling of visited lattice sites **/

/* Since the walker consumes the food it encounters, we need to keep
   track of the sites that have been visited (without being allowed
   to employ periodic boundary conditions). We borrow a standard approach
   from the simulation of self-avoiding walks, and store the visited
   sites in a hash table.
   To improve performance, a good hashing function (mapping the
   d-dimensional position to a single hash value) is required. We
   follow N. Madras and A. D. Sokal [J Stat Phys 50, 109 (1988)]:
   given the position x_k (k=1,...d), we calculate the hash as
   \sum_k a_k x_k (mod M) where M is the number of slots in the
   hash table ("buckets", although C++ handles the size of the hash
   in a more dynamic fashion). The a_k are chosen to be pairwise
   coprime and a_k = O(M^(k/(d+1))). Refer to Madras/Sokal for a
   discussion of this choice (aimed to minimize expensive hashing collisions).
   Compared to a naive hashing function (the default one used by the GNU
   implementation) I found the Madras/Sokal hash to be 15% to 20% faster
   for smallish runs, and about 50% faster after a few million steps in 2D
   for a case where the walks encounter many unique positions.
*/
// aimed-for size of the hash table
// for very long runs, we limit the (ab)use of RAM somewhat
// this ought to be std::min<unsigned long>(...), but we have to
// wait for C++14 for a constexpr min/max in the standard
constexpr static const unsigned long Nbuckets = (length*10 > MAX_BUCKETS ? MAX_BUCKETS : length*10);
// numbers a_k used for calculating the hash values
std::array<unsigned long,dim> hash_prime;
// implementation of the Madras/Sokal hash function
// works by specializing the boost::hash template
// replace the #if 1 by #if 0 to select the C++/Boost default hash
#if 1
namespace boost {
template <> struct hash <pos_t> {
  inline size_t operator () (const pos_t &pos) const {
    unsigned long h = 0;
    for (unsigned int d=0; d<dim; ++d) h += hash_prime[d]*pos[d];
    return h % Nbuckets;
  }
};
};
#endif
/* The Madras/Sokal hash needs a set of coprime numbers spread out
   evenly up to M. We calculate them on the fly. Since M can be quite
   large, calculating primes up to M is too costly. Luckily, we only need
   the a_k to be coprime, not actually prime themselves. We make use of the
   following results from number theory:
   - if p is prime, 2^p-1 is a pernicious Mersenne number
     (not necessarily itself prime, as e.g. 2^11-1 = 23*89)
   - the set of pernicious Mersenne numbers is pairwise coprime
   Since we store indices in (unsigned long) variables, b bits long, we do not
   need pernicious Mersenne numbers larger than 2^b-1. The list of
   primes between p=2 and p<=b is quickly calculated using the
   sieve of Erastosthenes.
*/
// implementation of the sieve of Erastosthenes
// C++ style implementation adapted from Loki Astari, see
// http://stackoverflow.com/questions/1954858/sieve-of-eratosthenes-algorithm
std::set<unsigned int> sieve (const unsigned int max) {
  std::set<unsigned int> sieve;
  for (unsigned int i=2; i < max; ++i) sieve.insert(i);
  for (auto i = sieve.begin(); i != sieve.end(); ++i) {
    unsigned int prime = *i;
    auto deleter = i; ++deleter;
    while (deleter != sieve.end()) {
      // the shorter original code (hg versions <= 6) simply used
      // while (++deleter != sieve.end()) { ... }
      // but it violates the C++ standard, since after an erase(),
      // the pointer is invalid; fixed by introducing the next variable
      // (removes complaints by valgrind)
      auto next = deleter; ++next;
      if (((*deleter) % prime)==0) sieve.erase(deleter);
      deleter = next;
    }
  }
  return sieve;
}
// initialize the a_k for the Madras/Sokal hash
void initialize_hash () {
  std::cerr << "seeking primes for " << Nbuckets << " buckets" << std::endl;
  // get a list of primes p
  std::set<unsigned int> prime_numbers = sieve (8*sizeof(unsigned long));
  // pick as many numbers as there are dimensions
  unsigned int knum = 0;
  double k_increment = 1./(dim+1);
  double current_k = k_increment;
  for (auto i = prime_numbers.begin(); i != prime_numbers.end(); ++i) {
    unsigned long coprime = (1<<(*i))-1;
    // pick 2^p-1 if its magnitude is close to the one desired
    if (log(double(coprime))/log(double(Nbuckets)) > current_k) {
      std::cerr << "prime " << coprime << " number " << knum << std::endl;
      hash_prime[knum] = coprime;
      current_k += k_increment;
      ++knum;
      if (knum>=dim) break;
    }
  }
  std::cerr << "primes done" << std::endl;
}


/** (6) global variables for handling the lattice **/

/* The d-dimensional square lattice has direction vectors (\pm1,0,0,...),
   (0,\pm1,0,0,...), ... (0,0...,0,\pm 1). These we store in a global
   array variable direction. Hence, once we have picked a random direction
   k = 0,...2d-1, we can say sth. like walkpos = currentpos + direction[k].
   We also offer a place to store cumulative probabilities for choosing
   direction number k, although this is currently not needed (see below).
*/
static std::array<pos_t,dim*2> direction;
static std::array<double,dim*2> accprob;
// alternative layout: store direction[k]=d, direction_sign[k]=+/-1
// and update position by walkpos[direction[k]] += direction_sign[k]
// At least up to d=8 this does not appear to be faster.


/** (7) global variables storing the results **/

/* We calculate the MSD and the number of distinct sites visited.
   (The latter is the average number of sites visited by a single walk,
   which is somewhat different from the number of sites visited by
   N walkers.) We also keep track of the total amount of food consumed.
   The van Hove function can be calculated at selected times (see above).
*/
// These variables are initialized in initialize() and filled by run().
// Since we only calculate their values on a semi-logarithmic blocked grid
// with blocks of length B, we do not need length places to store,
// but only reslength = (floor(log2(length/B))+1)*B
constexpr static unsigned long reslength = (_log2_<length/B>::apply()+1)*B;
struct results {
  std::array <double,reslength> times;
  std::array <double,reslength> msd;
  std::array <unsigned long,reslength> distinct_sites;
  std::array <unsigned int,reslength> filled_neighbors;
  std::array <double,reslength> food_consumed;
  #if COUNT_BACKBONE
  std::array <unsigned int,reslength> backbone_count;
  #endif
  //std::array <double,reslength> available_neighbors;
  // van Hove function:
  // for each time, store a pair (time,array of bins)
  typedef std::array<unsigned int,2*vanhove_cutoff+1> vanhove_bins_t;
  std::vector<std::pair<unsigned long,vanhove_bins_t> > vanhove;
  // init by constructor
  inline results () {
    for (unsigned long i = 0; i<reslength; ++i) times[i] = 0.0;
    for (unsigned long i = 0; i<reslength; ++i) msd[i] = 0.0;
    for (unsigned long i = 0; i<reslength; ++i) distinct_sites[i] = 0;
    for (unsigned long i = 0; i<reslength; ++i) filled_neighbors[i] = 0;
    for (unsigned long i = 0; i<reslength; ++i) food_consumed[i] = 0.0;
    #if COUNT_BACKBONE
    for (unsigned long i = 0; i<reslength; ++i) backbone_count[i] = 0.0;
    #endif
    //for (unsigned long i = 0; i<reslength; ++i) available_neighbors[i] = 0;
    if (calc_vanhove) {
      std::vector<unsigned long> vanhove_times = VANHOVE_TIMES;
      for (auto time : vanhove_times) {
        vanhove.push_back (std::make_pair(time,vanhove_bins_t()));
      }
    }
  }
  inline void add (const results &r) {
    for (unsigned long i = 0; i<reslength; ++i) times[i] = r.times[i];
    #ifndef NO_AVERAGES
    for (unsigned long i = 0; i<reslength; ++i) msd[i] += r.msd[i];
    for (unsigned long i = 0; i<reslength; ++i)
      distinct_sites[i] += r.distinct_sites[i];
    for (unsigned long i = 0; i<reslength; ++i)
      filled_neighbors[i] += r.filled_neighbors[i];
    for (unsigned long i = 0; i<reslength; ++i)
      food_consumed[i] += r.food_consumed[i];
    //for (unsigned long i = 0; i<reslength; ++i)
    //  available_neighbors[i] += r.available_neighbors[i];
    #if COUNT_BACKBONE
    for (unsigned long i = 0; i<reslength; ++i)
      backbone_count[i] += r.backbone_count[i];
    #endif
    #else
    for (unsigned long i = 0; i<reslength; ++i) msd[i] = r.msd[i];
    for (unsigned long i = 0; i<reslength; ++i)
      distinct_sites[i] = r.distinct_sites[i];
    for (unsigned long i = 0; i<reslength; ++i)
      filled_neighbors[i] = r.filled_neighbors[i];
    for (unsigned long i = 0; i<reslength; ++i)
      food_consumed[i] = r.food_consumed[i];
    //for (unsigned long i = 0; i<reslength; ++i)
    //  available_neighbors[i] = r.available_neighbors[i];
    #if COUNT_BACKBONE
    for (unsigned long i = 0; i<reslength; ++i)
      backbone_count[i] = r.backbone_count[i];
    #endif
    #endif
  }
};
results globals;



/** (8) run-time initialization **/

// we assume later on that this code has been called, otherwise
// undefined behavior may result
void initialize () {
  rng.seed (rng_seed); // seed the RNG
  initialize_hash ();  // initialize the hashing of visited sites

#if 0
  for (unsigned long i = 0; i<reslength; ++i) times[i] = 0.0;
  for (unsigned long i = 0; i<reslength; ++i) msd[i] = 0.0;
  for (unsigned long i = 0; i<reslength; ++i) distinct_sites[i] = 0;
  for (unsigned long i = 0; i<reslength; ++i) filled_neighbors[i] = 0;
  for (unsigned long i = 0; i<reslength; ++i) food_consumed[i] = 0.0;
#endif

  // setup the d-dimensional square lattice
  // direction[2k]   = (0,0,...,+1,0,....)
  // direction[2k+1] = (0,0,...,-1,0,....)
  // where the +/-1 appears in the k-th place (indexing from zero)
  for (unsigned int k = 0; k<dim; ++k) {
    for (unsigned int d = 0; d<dim; ++d) {
      direction[2*k][d] = 0;
    }
    direction[2*k][k] = 1;
    for (unsigned int d = 0; d<dim; ++d) {
      direction[2*k+1][d] = 0;
    }
    direction[2*k+1][k] = -1;
  }
  // setup cumulative probabilities for deciding where an unbiased
  // walker shall go
  // this is currently unused
  double acc_prob = 0;
  for (unsigned int k = 0; k<dim*2; ++k) {
    acc_prob += 1./(dim*2);
    accprob[k] = acc_prob;
  }

#if 0
  // initialize calculation of van Hove functions
  if (calc_vanhove) {
    std::vector<unsigned long> vanhove_times = VANHOVE_TIMES;
    for (auto time : vanhove_times) {
      vanhove.push_back (std::make_pair(time,vanhove_bins_t()));
    }
  }
#endif

  std::cout.setf(std::ios::scientific);
}


/** (9) food **/

/* To bias the walker, we provide a "food supply" F on each lattice site.
   If a walker sits at position x, the probabilities to take a step dx
   (where dx is one of the lattice vectors) is proportional to
   exp(F(x+dx)). The walker consumes all the food it finds on its
   lattice site after having stepped there.
   To implement F(x), we implement classes derived from food_declaration
   that implement an operator(). Food consumption is /not/ modeled here,
   and the operator() will only be called for sites where food has not yet
   been consumed.
*/

// The base class food_declaration declares the interface we need:
// we need an operator() implementing the actual food values (taking
// a pos_t vector indicating the position of the walker), and, for
// printing file headers, we need an info() method.
// The operator() is performance critical, and therefore we should not
// implement it as a virtual function (calls should not have the overhead
// of virtual-function resolution and should be inlined by the compiler).
// To get both compile-time optimization and the abstraction and type safety
// of class inheritance, we use the curiously recurring template (CRT)
// pattern, a standard C++ programming idiom.
template <typename food_impl>
class food_declaration {
public:
  inline double operator () (const pos_t &pos) const {
    return static_cast<const food_impl*>(this)->operator () (pos);
  }
  inline double exp (const pos_t &pos) const {
    return static_cast<const food_impl*>(this)->exp (pos);
  }
  virtual std::ostream &info (std::ostream &) const = 0;
};

/* example food implementations */

/* Constructor calls to derived classes following here can be used
   in place of the default FOOD preprocessor macro definition. */

// level_food(F0): create a homogeneous food supply of strength F0
class level_food : public food_declaration<level_food> {
  double f0, expf0;
public:
  inline level_food (double val) : f0(val) { expf0 = std::exp(val); }
  inline double operator () (const pos_t &) const { return f0; }
  inline double exp (const pos_t &) const { return expf0; }
  virtual std::ostream &info (std::ostream &os) const {
    return os << "# homogeneous food supply, F0 = " << f0 << std::endl;
  }
};

// step_food(F0): fill the space defined by all Cartesian x_i>0 with F0
class step_food : public food_declaration<step_food> {
  double f0, expf0;
  inline bool inside (const pos_t &pos) const {
    bool in = true;
    for (unsigned int d=0; d<dim; ++d) {
      if (pos[d] <= 0) in = false;
    }
    return in;
  }
public:
  inline step_food (double val) : f0(val) { expf0 = std::exp(val); }
  inline double operator () (const pos_t &pos) const {
    return inside(pos) ? f0 : 0.0;
  }
  inline double exp (const pos_t &pos) const {
    return inside(pos) ? expf0 : 1.0;
  }
  virtual std::ostream &info (std::ostream &os) const {
    return os << "# homogeneous food on positive orthant, F0 = " << f0
      << std::endl;
  }
};

// food_packet(F0,length): create a length^dim cube around the origin
// of strength F0
class food_packet : public food_declaration<food_packet> {
  double f0, expf0;
  double length,l2;
  inline bool inside (const pos_t &pos) const {
    bool in = true;
    for (unsigned int d=0; d<dim; ++d) {
      if (pos[d] > l2 || pos[d] < -l2) in = false;
    }
    return in;
  }
public:
  inline food_packet (double val, double len) : f0(val),length(len) {
    expf0 = std::exp(val);
    l2 = length/2.0;
  }
  inline double operator () (const pos_t &pos) const {
    return inside(pos) ? f0 : 0.0;
  }
  inline double exp (const pos_t &pos) const {
    return inside(pos) ? expf0 : 1.0;
  }
  virtual std::ostream &info (std::ostream &os) const {
    return os << "# confined food supply, F0 = " << f0
              << ", box of length L = " << length << std::endl;
  }
};


/** (10) percolating matrix **/

/* We allow the walker to be constrained to a pre-defined set of allowed
   sites which we assume to be percolating. The class percolating_matrix
   implemented here serves as an interface allowing to initialize such
   a "matrix" and to define an API for the random-walk algorithm to request
   whether a given move can be accepted or not.
   The simplest approach is to store a matrix of randomly occupied sites,
   and to provide an algorithm figuring out the (largest) percolating
   cluster. This quickly becomes unfeasible in large dimensions, where an
   adapted scheme trying to build just one percolating cluster has to be
   used. In order to allow for flexibility in choosing the percolating-cluster
   implementation, we construct different "engines" to be used as
   specializations of the percolating_matrix class.
*/

// The base class percolating_matrix is again implemented using the
// CRT idiom. It serves to define the API, involving:
// - the shape of the matrix (Lx * Ly * Lz ...), currently Lx=Ly=Lz=...
// - initialization, taking the occupation probability p as a parameter
// - a boolean test allowed(pos), indicating acceptance of a move to pos
template <typename engine>
class percolating_matrix {
protected:
  std::array<unsigned long,dim> shape;
public:
  // initialization for a matrix of box length Lx (all dimensions)
  // ensures that each length of the box is at least 1
  // if fewer lengths than dimensions are given, fill the lengths along
  // the innermost dimensions with the last length given
  inline percolating_matrix<engine> () {
    unsigned long lastLx = 1;
    for (unsigned int d=0; d<dim; ++d) {
      if (Lx[d]>0) lastLx = Lx[d];
      shape[d] = Lx[d] ? Lx[d] : lastLx;
    }
  }
  // further initialization is delegated to the engine implementation
  inline void init () {
    static_cast<engine*>(this)->init();
  }
  // this is meant to setup the percolating cluster information
  // with a given probability p for sites to be occupied
  inline void realize (double p) {
    static_cast<engine*>(this)->realize(p);
  }
  // test for allowed sites: delegated to the engine implementation
  inline bool allowed (const pos_t &pos) const {
    return static_cast<const engine*>(this)->allowed(pos);
  }
  // test whether a site is on the backbone, delegated to engine
  inline bool on_backbone (const pos_t &pos) const {
    return static_cast<const engine*>(this)->on_backbone(pos);
  }
  // virtual method to print file-header information
  virtual std::ostream &info (std::ostream &) const = 0;
  // wrapper to extract length information
  inline unsigned long L (unsigned int d) const {
    return static_cast<const engine*>(this)->L_impl(d);
  }
  inline unsigned long L_impl (unsigned int d) const { return shape[d]; }
};

/* percolating-matrix engines */

// empty_space: does what it says... all sites are allowed
// If using this specialization, the compiler should do its job and
// optimize away all calls to allowed(). In effect, the code should be
// as efficient with this engine in place, as for the simulation of a free walk.
class empty_space : public percolating_matrix<empty_space> {
public:
  inline void init () {}
  inline void realize (double) {}
  inline bool allowed (const pos_t &) const { return true; }
  inline bool on_backbone (const pos_t &) const { return true; }
  virtual std::ostream &info (std::ostream &os) const { return os; }
  inline unsigned long L_impl (unsigned int d) const { return 0; }
};

// hoshen_kopelman: Implement a randomly filled matrix and the
// detection of a percolating cluster by the Hoshen-Kopelman algorithm
// [https://www.ocf.berkeley.edu/~fricke/projects/hoshenkopelman/hoshenkopelman.html]
// The algorithm works as follows:
// A matrix is filled with values >0 and 0 randomly, indicating occupied and
// empty sites. The walker will only be allowed to step on occupied sites.
// (This hence implements site percolation.)
// The values stored in the matrix are used to label individual clusters,
// and the initial position of the walker is chosen to be on the largest
// percolating cluster.
template <bool allcluster,bool using_backbone>
class hoshen_kopelman_base :
public percolating_matrix<hoshen_kopelman_base<allcluster,using_backbone> > {
  typedef
    percolating_matrix<hoshen_kopelman_base<allcluster,using_backbone> > base;
  // choose the smallest integer type that will be able to keep all labels
  // on the conservative assumption that we might need up to L^dim ones
  // (we calculate 2^(dim*log2(LX))=LX^dim and its log2 to get the number
  // of bits required to store LX^dim labels; this might of course fail
  // for highly non-cubic matrices)
  constexpr static int labelsize =
    (_log2_<(1l<<(dim*_log2_<long(LX)>::apply()))>::apply()+1) > 32 ? 64 : 32;
  typedef int_with_bits<labelsize>::type label_t;
  // the matrix is stored here:
  typedef boost::multi_array<label_t,dim> array_type;
  std::unique_ptr<array_type> array_ptr;
  typedef typename std::array<array_type::index,dim> idx_t;
  unsigned long percolating_label, percsize;
  //
  mutable pos_t pbcwrap;
  //
  // The Hoshen-Kopelman algorithm works by sorting site labels into
  // equivalence classes: if a site is found that has neighbors belonging
  // to two different previously identified clusters, the labels identifying
  // these two clusters are marked as labeling the same cluster.
  // The equivalence classes are stored in the array labels, where each
  // entry contains the label of either the chosen representative of the
  // equivalence class, or another label in the same class.
  // During initialization, this is set up to be the identity mapping.
  std::vector<label_t> labels;
  // same_as(l) returns a representative of the equivalence class of l
  // Queries the labels array repeatedly, until a representative is found.
  // update v31: modifies the labels array to avoid nested links
  //   this seems to work fine, and dramatically increases performance!
  int same_as (label_t l) {
    unsigned long l0 = labels.size();
    label_t lnew;
    if (l>=l0) { // automatic resizing of label set
      labels.resize(l+1);
      for (label_t ll=l0; ll<=l; ++ll) labels[ll]=ll;
    }
    //while (l != labels[l]) l = labels[l];
    while (l != labels[l]) {
      lnew = labels[l]; labels[l] = labels[lnew]; l = lnew;
    }
    return l;
  }
  // link(l1,l2) joins two equivalence classes, modifying the labels array.
  inline void link (label_t l1, label_t l2) { labels[same_as(l1)]=same_as(l2); }
  // Main part of the Hoshen-Kopelman algorithm:
  // Assumes the matrix to be already generated, walks through all sites
  // fills the matrix with distinct labels for distinct clusters. At the
  // end, two sites belong to the same cluster iff they carry the same label.
  bool identify_clusters () {
    array_type &A = *array_ptr;
    // setup the label equivalence classes: initially, each label
    // only is its own representative
    for (unsigned long l=0; l<labels.size(); ++l) labels[l]=l;
    // stepping through a multi-dimensional matrix:
    // the dim numbers to address one element are stored in idx
    // we loop for the number of elements, increasing the last entry
    // of idx (innermost dimension) as we go, and take care of wrapping
    // around to the outer dimensions in idx by hand (see end of loop)
    idx_t idx; idx.fill(0);
    std::set<label_t> percolating_labels;
    std::vector<std::pair<label_t,label_t> > labels_to_be_linked;
    label_t label=1;
    for (unsigned long e=0; e<A.num_elements(); ++e) {
      // work on occupied matrix sites only:
      if (A(idx)>0) {
        // get indices belonging to "above" ("left") in each dimension
        // If all the neighbors have labels 0, the current site starts
        // a new cluster (with a new label >=2).
        // If a neighbor is found, the current site inherits
        // its label (neighbor_label). If a further neighbor is found,
        // its label is put into the same equivalence class as the first
        // neighbor's label.
        // Our cluster labels are >=2 so that we can perform in-place
        // rewriting of the 1's used to mark any occupied site with
        // cluster labels. (In fact, it should also work when starting
        // cluster labels with 1, but debugging is harder this way.)
        std::array<int,dim> upper_neighbors;
        label_t neighbor_label = 0;
        for (unsigned int d=0; d<dim; ++d) {
          if (idx[d]>0) {
            idx_t idx1 (idx);
            idx1[d]--;
            upper_neighbors[d] = A(idx1);
          } else {
            upper_neighbors[d] = 0;
          }
          // if all works as intended, we traverse the matrix such that
          // all "upper" neighbors have been touched already, and are either
          // empty or carry a cluster label >1
          if (upper_neighbors[d]==1) throw; // should never happen
          if (upper_neighbors[d]>1) {
            // the current site is occupied, and a neighbor is occupied
            // if this happens for more than one dimension, we can
            // link the labels of those neighbors as belonging to one cluster
            if (!neighbor_label) neighbor_label = upper_neighbors[d];
            else link (upper_neighbors[d],neighbor_label);
          }
        }
        if (neighbor_label) {
          A(idx) = same_as(neighbor_label);
        } else {
          ++label;
          A(idx) = label;
        }
      }
      // step the multi-index to traverse all matrix entries:
      int current_d = dim-1;
      while (++idx[current_d]>=base::shape[current_d]) {
        idx[current_d] = 0;
        if (current_d>0) --current_d;
      }
    }
    std::cerr << "cluster-find used labels up to " << label << std::endl;
    // perform p.b.c. checking:
    // If a cluster appears both at the upper and the lower boundary along
    // a dimension, it is spanning the box. If it touches itself across
    // a periodic boundary, it is also wrapping. We seek those clusters.
    // As of above, labels have not been identified across p.b.c.,
    // and this is important: having a same_as() connection requires a
    // chain of linked sites throughout the box! Otherwise (including
    // periodic-image neighbors in the above loop), the test performed here
    // would merely detect all clusters that happen to lie across a periodic
    // boundary.
    // For this reason, the final linking of p.b.c.-wrapping neighbors
    // has to be performed in a second pass.
    for (unsigned int d=0; d<dim; ++d) {
      idx.fill(0);
      for (unsigned long e=0; e<A.num_elements()/base::shape[d]; ++e) {
        int current_d = dim-1;
        if (current_d == d && current_d>0) --current_d;
        std::array<array_type::index,dim> idxwrap (idx);
        idxwrap[d] = base::shape[d]-1;
        if (A(idx)>1) {
          if (same_as(A(idx))==same_as(A(idxwrap))) {
            // we have found two labels that we are going to identify
            // since they touch across p.b.c.
            // if this happens now, it must be that the cluster percolates
            percolating_labels.insert(same_as(A(idx)));
          } else if (A(idxwrap)>1) {
            labels_to_be_linked.push_back (std::make_pair(A(idx),A(idxwrap)));
          }
        }
        // step multi-index (same as above),
        // but only in dim-1 dimensions to stay at the boundary
        while (++idx[current_d]>=base::shape[current_d]) {
          idx[current_d]=0;
          if (current_d>0) --current_d;
          if (current_d==d && current_d>0) --current_d;
        }
      }
    }
    // now, link the remaining labels across p.b.c.
    for (auto l : labels_to_be_linked) link (l.second,l.first);
    // go through the matrix again, replacing all labels by the
    // same representative of the equivalence classes
    std::vector<unsigned long> sizes (label+1);
    auto elements = boost::make_iterator_range (A.data(),
                                                A.data()+A.num_elements());
    for (unsigned long l=0; l<labels.size(); ++l) labels[l] = same_as(l);
    for (auto &element : elements) {
      if (element) element = same_as(element);
      sizes[element]++;
    }
    // figure out the largest percolating cluster
    percsize = 0;
    percolating_label = 0;
    for (auto l : percolating_labels) {
      unsigned long ll = same_as(l);
      if (sizes[ll]>percsize) { percsize=sizes[ll]; percolating_label=ll; }
    }
    if (percsize>0)
      std::cerr << "percolating cluster size " << percsize << std::endl;
    return (percsize>0);
  }
  // generate a randomly filled matrix
  // assumes memory for the matrix to be already allocated(!)
  // updated in v27: calculates a deterministic target for the number
  // of occupied lattice sites; the created matrices will always be filled
  // with probability p-1/L^d < p_num <= p.
  void generate_matrix (double p) {
    array_type &A = *array_ptr;
    // figure out how many matrix elements we want to be occupied in the end
    // if p*L^d is integer, the target can be exactly reached
    // otherwise we create a matrix that is slightly less filled than desired
    unsigned long target = p*A.num_elements();
    // make matrix elements == 1 with probability p, zero else
    auto elements = boost::make_iterator_range (A.data(),
                                                A.data()+A.num_elements());
    for (auto &element : elements) {
      element = random_number() < p ? 1 : 0;
    }
    // count the number of actual occupied sites in the matrix
    // This value divided by the total number of elements should approximate p.
    // The counting could be implemented in the above loop, but I wanted to
    // learn the C++11 syntax for for_each loops :-)
    // The (int val) { ...} syntax is an inline functor used by for_each
    // (C++11 lambda notation for functions). The [&n] in front indicates
    // that during the function call, the outer-scope variable n should
    // be "captured" (made accessible by reference).
    long n=0;
    std::for_each (elements.begin(), elements.end(),
      [&n](int val) { n+=val; });
    std::cerr << "matrix initially filled with " << n << " sites ("
      << double(n)/A.num_elements()*100 << "%)" << std::endl;
    while (n!=target) {
      unsigned long site = random_number()*A.num_elements();
      // not sure what to do here: could randomly flip sites
      // and trust that a random walk will sooner (or later...)
      // hit any desired number?
      //   if (A(site)==1) n--; else n++;
      //   A(site) = 1 - A(site);
      // what we do instead is a biased walk:
      if (n>target && A.data()[site]==1) { A.data()[site]=0; n--; }
      else if (n<target && A.data()[site]==0) { A.data()[site]=1; n++; }
    }
    std::cerr << "matrix finally filled with " << n << " sites ("
      << double(n)/A.num_elements()*100 << "%)" << std::endl;
  }
  // percolating backbone, to be allocated and filled by find_backbone()
  std::unique_ptr<array_type> backbone_ptr;
  #ifdef BACKBONE_EXP
  // subroutine used by ientify_backbone() to remove a dangling end
  inline void remove_dangling_end (std::pair<idx_t,unsigned long> i,
    array_type &Achem, bool wrap) {
    #ifdef DEBUG_CHEMDIST
    std::cout << "remove_dangling_end " << i.first << " " << wrap << std::endl;
    #endif
    array_type &allowed = *backbone_ptr;
    std::queue<std::pair<idx_t,unsigned long> > deleter_fifo;
    deleter_fifo.push(i);
    while (!deleter_fifo.empty()) {
      std::pair<idx_t,unsigned long> del = deleter_fifo.front();
      deleter_fifo.pop();
      if (!allowed(del.first)) continue;
      bool is_leaf = true;
      std::queue<std::pair<idx_t,unsigned long> > neighborlist;
      for (unsigned int d=0; d<dim; ++d) {
        idx_t nn (del.first);
        bool consider = true;
        if (nn[d]>0) nn[d]--;
        else { if (wrap) nn[d]=base::shape[d]-1; else consider=false; }
        if (consider && allowed(nn)) {
          if (Achem(nn)==del.second-1) {
            neighborlist.push(std::make_pair(nn,del.second-1));
          } else {
            is_leaf = false;
          }
        }
        nn=del.first;
        consider = true;
        if (nn[d]<base::shape[d]-1) nn[d]++;
        else { if (wrap) nn[d]=0; else consider=false; }
        if (consider && allowed(nn)) {
          if (Achem(nn)==del.second-1) {
            neighborlist.push(std::make_pair(nn,del.second-1));
          } else {
            is_leaf = false;
          }
        }
      }
      if (is_leaf) {
        allowed(del.first) = 0;
        while (!neighborlist.empty()) {
          deleter_fifo.push(neighborlist.front());
          neighborlist.pop();
        }
      }
    }
  }
  #endif
  // Try to identify the percolating backbone
  // Requires the percolating matrix to be already generated.
  void find_backbone () {
    const array_type &A = *array_ptr;
    // allocate a matrix storing backbone yes/no information
    // this can be used by other methods
    backbone_ptr = std::unique_ptr<array_type>(new array_type (base::shape));
    array_type &allowed = *backbone_ptr;
    // allocate a matrix storing chemical distances
    // for internal use, hence no need for a unique_ptr
    array_type *Achem_ptr = new array_type (base::shape);
    array_type &Achem = *Achem_ptr;

    for (unsigned long e=0; e<array_ptr->num_elements(); ++e) {
      if (array_ptr->data()[e] == percolating_label)
        backbone_ptr->data()[e] = 1;
    }
    // pick a center site
    idx_t i0;
    for (unsigned int d=0; d<dim; ++d) i0[d]=base::shape[d]/2;
    while (A(i0)!=percolating_label) {
      unsigned int k = random_number()*2*dim;
      i0 = i0 + direction[k];
    }
    #ifdef BACKBONE_EXP
    identify_backbone (i0,1,Achem);
    #ifdef DEBUG_CHEMDIST
      // debugging output (works in 2D)
      { idx_t idx; idx.fill(0);
      for (unsigned long ii=0; ii<base::shape[0]; ++ii) {
        for (unsigned long jj=0; jj<base::shape[1]; ++jj) {
          idx[0] = ii; idx[1] = jj;
          std::cout << (allowed(idx)?((0)?'*':'['):' ')
                    << std::setw(4) << Achem(idx)
                    << (allowed(idx)?((0)?'*':']'):' ');
        }
        std::cout << std::endl;
      }
        std::cout << std::endl;
      }
    #endif
    #ifdef PRINT_WALK
      { idx_t idx; idx.fill(0);
      for (unsigned long e=0; e<allowed.num_elements(); ++e) {
        if (allowed(idx)>0)
          std::cout << "B " << idx << " " << allowed(idx) << std::endl;
        // step multi-index
        int current_d = dim-1;
        while (++idx[current_d]>=base::shape[current_d]) {
          idx[current_d] = 0;
          if (current_d>0) --current_d;
        }
      }
      }
    #endif
    delete Achem_ptr;
  }
  void identify_backbone (idx_t i0, unsigned long l0,
    array_type &Achem, bool keep_boundary_nodes=true) {
    array_type &allowed = *backbone_ptr;
    std::queue<std::pair<idx_t,unsigned long> > fifo;
    fifo.push(std::make_pair(i0,l0));
    // grow layers of increasing chemical distance
    std::queue<std::pair<idx_t,unsigned long> > candidate_fifo;
    while (!fifo.empty()) {
      std::pair<idx_t,unsigned long> i = fifo.front(); fifo.pop();
      if (Achem(i.first)>0) continue;
      Achem(i.first)=i.second;
      #ifdef DEBUG_CHEMDIST
      // debugging output (works in 2D)
      { idx_t idx; idx.fill(0);
      for (unsigned long ii=0; ii<base::shape[0]; ++ii) {
        for (unsigned long jj=0; jj<base::shape[1]; ++jj) {
          idx[0] = ii; idx[1] = jj;
          std::cout << (allowed(idx)?((idx==i.first)?'*':'['):' ')
                    << std::setw(4) << Achem(idx)
                    << (allowed(idx)?((idx==i.first)?'*':']'):' ');
        }
        std::cout << std::endl;
      }
        std::cout << std::endl;
      }
      #endif
      bool keep = false;
      bool wrapping_candidate = false;
      for (unsigned int d=0; d<dim; ++d) {
        idx_t nn (i.first);
        //if (nn[d]>0) nn[d]--; else nn[d]=base::shape[d]-1;
        if (nn[d]>0) {
          nn[d]--;
          if (allowed(nn) && Achem(nn)==0) {
            fifo.push(std::make_pair(nn,i.second+1));
            keep = true;
          }
        } else {
          nn[d]=base::shape[d]-1;
          // on the boundary: if there is a site on the other side,
          // and that site has been seen, its not a dangling end
          // if there is a site that has not been seen: its a candidate
          // for wrapping, but may turn out to be a dangling end later
          if (allowed(nn)) {
            if (keep_boundary_nodes) {
              keep = true;
              if (Achem(nn)==0) candidate_fifo.push(i);
            } else if (Achem(nn)==0) {
              fifo.push(std::make_pair(nn,i.second+1));
              keep = true;
            }
          }
        }
        nn=i.first;
        //if (nn[d]<base::shape[d]-1) nn[d]++; else nn[d]=0;
        if (nn[d]<base::shape[d]-1) {
          nn[d]++;
          if (allowed(nn) && Achem(nn)==0) {
            fifo.push(std::make_pair(nn,i.second+1));
            keep = true;
          }
        } else {
          nn[d]=0;
          if (allowed(nn)) {
            if (keep_boundary_nodes) {
              keep = true;
              if (Achem(nn)==0) candidate_fifo.push(i);
            } else if (Achem(nn)==0) {
              fifo.push(std::make_pair(nn,i.second+1));
              keep = true;
            }
          }
        }
      }
      if (!keep) remove_dangling_end (i,Achem,true);
    }
    // go again through all boundary nodes that we have kept
    // if they are linked across pbc to something that we have also kept,
    // that's fine, else delete all those that are linked to sites we
    // have not visited from the other side
    while (!candidate_fifo.empty()) {
      std::pair<idx_t,unsigned long> i = candidate_fifo.front();
      for (unsigned int d=0; d<dim; ++d) {
        idx_t nn (i.first);
        bool on_boundary=false;
        if (nn[d]==0) {
          nn[d] = base::shape[d]-1; on_boundary=true;
        } else if (nn[d]==base::shape[d]-1) {
          nn[d] = 0; on_boundary=true;
        }
        if (on_boundary) {
          if (allowed(nn) && Achem(nn)==0) {
            // haven't seen the other side yet
            // we have to start the backbone identification here
            // and see what happens
            // the bb identification would see our site as a visited site
            // and use that as a criterion to definitely keep - trick the
            // algorithm into temporarily believing we're not there
            unsigned long l = Achem(i.first);
            //allowed(i.first) = 0;
            #ifdef DEBUG_CHEMDIST
              std::cout << "recurse " << nn << std::endl;
            #endif
            identify_backbone (nn,i.second+1,Achem,true);
            //allowed(i.first) = 1;
            // if the recursive call left the site, we connect, else...
            if (!allowed(nn)) remove_dangling_end (i,Achem,true);
          }
        }
      }
      candidate_fifo.pop();
    }
    #else
    std::queue<std::pair<idx_t,unsigned long> > fifo;
    fifo.push(std::make_pair(i0,1));
    // testing...:
    // pick boundary sites also as initial sites
    for (unsigned int d=0; d<dim; ++d) {
      i0.fill(0);
      for (unsigned int x=0; x<base::shape[d]; ++x) {
        i0[d]=x;
        if (A(i0)==percolating_label) fifo.push(std::make_pair(i0,1));
      }
    }
    // grow layers of increasing chemical distance
    while (!fifo.empty()) {
      std::pair<idx_t,unsigned long> i = fifo.front(); fifo.pop();
      if (Achem(i.first)>0) continue;
      Achem(i.first)=i.second;
      #ifdef DEBUG_CHEMDIST
      // debugging output (works in 2D)
      { idx_t idx; idx.fill(0);
      for (unsigned long ii=0; ii<base::shape[0]; ++ii) {
        for (unsigned long jj=0; jj<base::shape[1]; ++jj) {
          idx[0] = ii; idx[1] = jj;
          std::cout << (allowed(idx)?((idx==i.first)?'*':'['):' ')
                    << std::setw(4) << Achem(idx)
                    << (allowed(idx)?((idx==i.first)?'*':']'):' ');
        }
        std::cout << std::endl;
      }
      }
      #endif
      bool pushed_neighbor = false;
      for (unsigned int d=0; d<dim; ++d) {
        idx_t nn (i.first);
        if (nn[d]>0) nn[d]--; else nn[d]=base::shape[d]-1;
        if (allowed(nn) && Achem(nn)==0) {
          fifo.push(std::make_pair(nn,i.second+1));
          pushed_neighbor = true;
        }
        nn=i.first;
        if (nn[d]<base::shape[d]-1) nn[d]++; else nn[d]=0;
        if (allowed(nn) && Achem(nn)==0) {
          fifo.push(std::make_pair(nn,i.second+1));
          pushed_neighbor = true;
        }
      }
      if (!pushed_neighbor) {
        bool on_boundary = false;
        for (unsigned int d=0; !on_boundary && d<dim; ++d) {
          if (i.first[d]==0 || i.first[d]==base::shape[d]-1) on_boundary=true;
        }
        if (!on_boundary) {
          std::queue<std::pair<idx_t,unsigned long> > deleter_fifo;
          deleter_fifo.push(std::make_pair(i.first,i.second));
          while (!deleter_fifo.empty()) {
            std::pair<idx_t,unsigned long> del = deleter_fifo.front();
            deleter_fifo.pop();
            if (!allowed(del.first)) continue;
            bool is_leaf = true;
            std::queue<std::pair<idx_t,unsigned long> > neighborlist;
            for (unsigned int d=0; d<dim; ++d) {
              idx_t nn (del.first);
              if (nn[d]>0) nn[d]--; else nn[d]=base::shape[d]-1;
              if (allowed(nn)) {
                if (Achem(nn)==del.second-1) {
                  neighborlist.push(std::make_pair(nn,del.second-1));
                } else {
                  is_leaf = false;
                }
              }
              nn=del.first;
              if (nn[d]<base::shape[d]-1) nn[d]++; else nn[d]=0;
              if (allowed(nn)) {
                if (Achem(nn)==del.second-1) {
                  neighborlist.push(std::make_pair(nn,del.second-1));
                } else {
                  is_leaf = false;
                }
              }
            }
            if (is_leaf) {
              allowed(del.first) = 0;
              while (!neighborlist.empty()) {
                deleter_fifo.push(neighborlist.front());
                neighborlist.pop();
              }
            }
          }
        }
      }
    }
    #ifdef DEBUG_CHEMDIST
      // debugging output (works in 2D)
      { idx_t idx; idx.fill(0);
      for (unsigned long ii=0; ii<base::shape[0]; ++ii) {
        for (unsigned long jj=0; jj<base::shape[1]; ++jj) {
          idx[0] = ii; idx[1] = jj;
          std::cout << (allowed(idx)?((idx==i.first)?'*':'['):' ')
                    << std::setw(4) << Achem(idx)
                    << (allowed(idx)?((idx==i.first)?'*':']'):' ');
        }
        std::cout << std::endl;
      }
      }
    #endif
    #ifdef PRINT_WALK
      { idx_t idx; idx.fill(0);
      for (unsigned long e=0; e<allowed.num_elements(); ++e) {
        if (allowed(idx)>0)
          std::cout << "B " << idx << " " << allowed(idx) << std::endl;
        // step multi-index
        int current_d = dim-1;
        while (++idx[current_d]>=base::shape[current_d]) {
          idx[current_d] = 0;
          if (current_d>0) --current_d;
        }
      }
      }
    #endif
    delete Achem_ptr;
    #endif
  }
public:
  inline hoshen_kopelman_base () : labels(Lx[0]) { pbcwrap.fill(0); }
  // Initialization: allocate memory for the matrix
  void init () {
    // calculate number of entries in the matrix
    // (we do not assume the matrix to be square)
    unsigned long num_elements = 1;
    for (unsigned int d=0; d<dim; ++d) num_elements *= base::shape[d];
    std::cerr << "allocating matrix (" << base::shape[0];
    if (dim>1)
      for (unsigned int d=1; d<dim; ++d) std::cerr << "x" << base::shape[d];
    std::cerr << "): "
      << memsize(num_elements*sizeof(array_type::element)) << std::endl;
    // create the matrix in memory
    // this could well fail due to an out-of-memory error...
    array_ptr = std::unique_ptr<array_type>(new array_type (base::shape));
  }
  // Create a realization of the randomly filled matrix.
  void realize (double p) {
    // we keep trying until a percolating cluster is generated;
    // if we fail after a maximum number of trials, abort
    bool percolates = false;
    int trial = 0;
    while (!percolates && ++trial < 100) {
      generate_matrix (p);
      if (!allcluster) {
        percolates = identify_clusters();
        if (!percolates) std::cerr << " ... does not percolate" << std::endl;
      } else break;
    }
    if (!allcluster) if (!percolates) throw;
    #ifdef PRINT_WALK
    std::array<array_type::index,dim> idx; idx.fill(0);
    for (unsigned long e=0; e<array_ptr->num_elements(); ++e) {
      if ((*array_ptr)(idx) == percolating_label) {
        std::cout << "C " << idx << std::endl;
      }
      // step the multi-index to traverse all matrix entries:
      int current_d = dim-1;
      while (++idx[current_d]>=base::shape[current_d]) {
        idx[current_d] = 0;
        if (current_d>0) --current_d;
      }
    }
    #endif
    if (!allcluster && (using_backbone || count_backbone)) find_backbone ();
  }

  // return true if the given position is on the percolating cluster
  // (or on the backbone, if this is our interpretation as set by
  // the class' template parameter)
  inline bool allowed (const pos_t &pos) const {
    pos_t idx = pos;
    // periodic-boundary wrapping is needed now
    // our best guess (based on previous calls) for the shifts
    // needed to wrap the position into the box is pbcwrap
    // if this doesn't do the job, we adjust both the position
    // and our guess for the shifts accordingly
    for (unsigned int d=0; d<dim; ++d) {
      idx[d] += pbcwrap[d];
      while (idx[d]<0) { idx[d] += base::shape[d]; pbcwrap[d] += base::shape[d]; }
      while (idx[d]>=base::shape[d]) { idx[d] -= base::shape[d]; pbcwrap[d] -= base::shape[d]; }
    }
    if (allcluster) return ((*array_ptr)(idx) != 0);
    return (using_backbone ? ((*backbone_ptr)(idx) != 0)
                           : ((*array_ptr)(idx) == percolating_label));
  }
  // return true if the given position is on the backbone of the cluster
  // this assumes that the backbone has been identified beforehand,
  // otherwise a call to this method will segfault!
  inline bool on_backbone (const pos_t &pos) const {
    pos_t idx = pos;
    // this uses the pbcwrap optimized by allowed()
    for (unsigned int d=0; d<dim; ++d) {
      idx[d] += pbcwrap[d];
      while (idx[d]<0) { idx[d] += base::shape[d]; }
      while (idx[d]>=base::shape[d]) { idx[d] -= base::shape[d]; }
    }
    return ((*backbone_ptr)(idx) != 0);
  }
  // header information
  virtual std::ostream &info (std::ostream &os) const {
    os << "# site-percolating cluster (HK algorithm, p = " << p << ")\n";
    if (allcluster) os << "# all-cluster average\n";
    if (using_backbone) os << "# backbone only\n";
    os << "# L = " << base::shape[0];
    if (dim>1) for (unsigned int d=1; d<dim; ++d) os << "x" << base::shape[d];
    os << "\tNmatrices = " << Nmatrices;
    os << std::endl;
  }
};

// shortcuts
typedef hoshen_kopelman_base<false,false> hoshen_kopelman;
typedef hoshen_kopelman_base<false,true>  hoshen_kopelman_bb;
typedef hoshen_kopelman_base<true,false> hoshen_kopelman_allcluster;

// at the request of a referee: what happens on the Sierpinski square?
// this code only has been tested for dim=2 and for LX=multiple of 9,
// so beware
// test case:
// make rwfood NRUNS=100 LENGTH=1000000 DIM=2 FOOD="level_food(F)" VANHOVE=false RNGSEED="time(0)" RANDOM_START=true NMATRICES=1 MATRIX=sierpinski_square LX=6561
class sierpinski_square : public percolating_matrix<sierpinski_square> {
  constexpr static int labelsize =
    (_log2_<(1l<<(dim*_log2_<long(LX)>::apply()))>::apply()+1) > 32 ? 64 : 32;
  typedef int_with_bits<labelsize>::type label_t;
  typedef boost::multi_array<label_t,dim> array_type;
  std::unique_ptr<array_type> array_ptr;
  typedef std::array<array_type::index,dim> idx_t;
  //
  mutable pos_t pbcwrap;
  //
  void block_center (idx_t offset, idx_t width) {
    idx_t blockwidth;
    unsigned long num_e = 1;
    bool recurse = false;
    for (unsigned int d=0; d<dim; ++d) {
      if (width[d]<3) return;
      blockwidth[d] = width[d]/3;
      num_e *= blockwidth[d];
      if (blockwidth[d]>=3) recurse = true;
    }
    idx_t idx = offset + blockwidth;
    for (unsigned long e=0; e<num_e; ++e) {
      (*array_ptr)(idx) = 1;
      int current_d = dim-1;
      if (num_e>1)
      while (++idx[current_d]>=offset[current_d]+2*blockwidth[current_d]) {
        idx[current_d] = offset[current_d]+blockwidth[current_d];
        if (current_d>0) --current_d;
      }
    }
    if (!recurse) return;
    idx_t block; block.fill(0);
    idx_t blockoffset;
    num_e = pow<dim>(3);
    for (unsigned int b=0; b<num_e; ++b) {
      bool center_block = true;
      for (unsigned int d=0; d<dim; ++d) {
        if (block[d]!=1) center_block = false;
        blockoffset[d] = block[d]*blockwidth[d];
      }
      if (!center_block) block_center (offset+blockoffset, blockwidth);
      int current_d = dim-1;
      while (++block[current_d]>=3) {
        block[current_d] = 0;
        if (current_d>0) --current_d;
      }
    }
  }
public:
  inline sierpinski_square () { pbcwrap.fill(0); }
  void init () {
    unsigned long num_elements = 1;
    for (unsigned int d=0; d<dim; ++d) {
      shape[d] = shape[0];
      num_elements *= shape[d];
    }
    std::cerr << "allocating matrix (" << shape[0];
    if (dim>1) for (unsigned int d=1; d<dim; ++d) std::cerr << "x" << shape[d];
    std::cerr << "): "
      << memsize(num_elements*sizeof(array_type::element)) << std::endl;
    // create the matrix in memory
    // this could well fail due to an out-of-memory error...
    array_ptr = std::unique_ptr<array_type>(new array_type (shape));
  }
  void realize (double) {
    idx_t idx0; idx0.fill(0);
    idx_t idx1; for (unsigned int d=0; d<dim; ++d) idx1[d] = shape[d];
    block_center (idx0,idx1);
#ifdef PRINT_WALK
    idx_t idx; idx.fill(0);
    for (unsigned long e=0; e<array_ptr->num_elements(); ++e) {
      std::cerr << ((*array_ptr)(idx)) << " ";
      // step the multi-index to traverse all matrix entries:
      int current_d = dim-1;
      bool newl = false;
      while (++idx[current_d]>=shape[current_d]) {
        if (!newl) std::cerr << std::endl; newl = true;
        idx[current_d] = 0;
        if (current_d>0) --current_d;
      }
    }
#endif
  }
  inline bool allowed (const pos_t &pos) const {
    pos_t idx = pos;
    // periodic-boundary wrapping is needed now
    // our best guess (based on previous calls) for the shifts
    // needed to wrap the position into the box is pbcwrap
    // if this doesn't do the job, we adjust both the position
    // and our guess for the shifts accordingly
    for (unsigned int d=0; d<dim; ++d) {
      idx[d] += pbcwrap[d];
      while (idx[d]<0) { idx[d] += shape[d]; pbcwrap[d] += shape[d]; }
      while (idx[d]>=shape[d]) { idx[d] -= shape[d]; pbcwrap[d] -= shape[d]; }
    }
    return ((*array_ptr)(idx) == 0);
  }
  inline bool on_backbone (const pos_t &) const { return true; }
  virtual std::ostream &info (std::ostream &os) const {
    os << "# Sierpinski square\n";
    os << "# L = " << shape[0];
    if (dim>1) for (unsigned int d=1; d<dim; ++d) os << "x" << shape[d];
    os << "\tNmatrices = " << Nmatrices;
    os << std::endl;
  }
};

/** (11) main procedure: food-biased random walk **/

/* run():
   - Creates a new position list and a new hash to count visited sites.
   - Takes length steps of the walker.
   - Calculates MSD and number of visited sites.
   Currently, the walker always starts at the origin, if no matrix is given.

   Compile-time switch: THREAD_RNG
   If set, run() creates its own rng, seeded with a combination of the global
   rng_seed and the value of runidx passed by the caller. As a consequence,
   several threads can call run() in parallel, without side-effects on the
   rng. If not set, calls to random_number made during run() refer to and
   advance the state of a single global rng; the output of the program will
   then depend on the number of concurrent threads.
*/

template <typename food_class, typename engine>
unsigned long run (const food_declaration<food_class> &food,
  const percolating_matrix<engine> &matrix,
  unsigned long runidx, results &res) {
  #ifdef THREAD_RNG
  mersenne_twister local_rng; // local rng object
  local_rng.seed (rng_seed + runidx);
  auto random_number = std::bind (uniform_distribution, std::ref(local_rng));
  #endif
  // easy and clean:
  //poslist_t position_list;
  // but this segfaults for large run lengths, because it tries to
  // create the array on the stack
  // solution: create it on the heap (by explicitly calling new)
#if 0
  std::unique_ptr<poslist_t> position_list_ptr (new poslist_t);
  poslist_t &position_list = *position_list_ptr;
#endif
  // same story for the hash
  // we could omit the size specification here to allow
  // for on-the-stack creation, with on the fly resizing;
  // this turns out to be a significant performance penalty for long runs!
  //pos_hash_t position_hash; // (Nbuckets);
  std::unique_ptr<pos_hash_t> position_hash_ptr (new pos_hash_t (Nbuckets));
  pos_hash_t &position_hash = *position_hash_ptr;
  #ifdef BIAS_NUM_VISITS
  std::unique_ptr<pos_count_t> position_count_ptr (new pos_count_t (Nbuckets));
  pos_count_t &position_count = *position_count_ptr;
  #endif
  #ifdef TEST_MULTI_EAT
    // (tv) 2018-03-07: suggestion by Peter Grassberger, 2018-02-23:
    // make the walker eat food also from adjacent sites
    // should perhaps now really introduce the possibility to keep
    // food pre-stored in an array
    std::unique_ptr<pos_hash_t> eaten_hash_ptr (new pos_hash_t (Nbuckets));
    pos_hash_t &eaten_hash = *eaten_hash_ptr;
  #else
    #define eaten_hash position_hash
  #endif
  pos_t startpos; startpos.fill(0);
  // optionally randomize the starting position
  // if random_start_center is set, we restrict to somewhere near the center
  // of the box, this might be needed when working with the backbone
  if (random_start) {
    for (unsigned int d=0; d<dim; ++d) {
      const unsigned int L0 = random_start_center ? matrix.L(d)/3 : matrix.L(d);
      startpos[d] = random_number()*L0
                  + (random_start_center ? matrix.L(d)/3 : 0);
    }
  }
  // adjust the starting position to be on the percolating cluster
  // if not chosen randomly, we just increase the starting x-pos
  // note that this might in rare circumstances hang (if the initial point
  // is one a percolating line along x of forbidden sites)
  while (!matrix.allowed(startpos)) {
    unsigned int k (random_start ? random_number()*2*dim : 0);
    startpos = startpos + direction[k];
  }
#if 0
  position_list[0] = startpos;

  pos_t walkpos = position_list[0];
#else
  pos_t walkpos = startpos;
#endif
  pos_t newpos;
  position_hash.insert(walkpos);
  #ifdef TEST_MULTI_EAT
    eaten_hash.insert(walkpos);
  #endif
  #ifdef BIAS_NUM_VISITS
  position_count[walkpos] = 1;
  #endif
  #ifdef PRINT_WALK
  std::cout << 0 << "\t" << walkpos << std::endl;
  #endif

  unsigned long num_distinct_sites = 0;
  unsigned long backbone_count = 0;
  double eaten = 0.0;
  bool accept_move = true;
  std::array<double,dim*2> current_accprob;
  unsigned long printi = 1, iinc = 1, blocki = 1, ti = 1;
  auto next_vh = res.vanhove.begin();
  // perform "length" steps for the random walker
  #ifdef TIMED_LOG
    auto lastlogtime = std::chrono::system_clock::now();
  #endif
  for (unsigned long i=1; i<length; ++i) {
    // pick a random number 0 <= r < 1 and find the lattice direction k such
    // that the cumulative probability obeys accprob[k-1] <= r < accprob[k]
    double r = random_number();
    // without food: can use pre-calculated global accprob
    // with food: calculate current_accprob on the fly
    double accumulated = 0.0;
    unsigned int filled = 0;
    //unsigned int available = 0;
    double food_at_site [dim*2];
    bool allowed [dim*2];
    for (unsigned int k=0; k<dim*2; ++k) {
      pos_t probe_position = walkpos + direction[k];
      allowed[k] = matrix.allowed(probe_position);
      //if (allowed[k]) ++available;
      #ifndef BIAS_NUM_VISITS
      if (eaten_hash.find (probe_position) == eaten_hash.end()
        #ifndef FOOD_ON_BLOCKED_SITES
          && allowed[k]
        #endif
      ) {
        // we've not yet been here, so there is food
        #ifdef FOOD_SWITCH
        if (i>FOOD_SWITCH_TIME) {
          food_at_site[k] = 0.0;
          accumulated += 1.0;
        } else {
        #endif
        food_at_site[k] = food(probe_position);
        accumulated += food.exp(probe_position);
        #ifdef FOOD_SWITCH
        }
        #endif
        filled++;
      } else {
        // we've been here, so there is no food left
        // or, there's no percolating site here, so we won't go here
        // to get correct random-walk statistics, we nevertheless
        // need to include the site here in calculated in the acc.prob.
        food_at_site[k] = 0.0;
        #ifdef MYOPIC_ANT
        if (allowed[k])
        #endif
        accumulated += 1.0; // exp(0)
      }
      #else
      if (allowed[k]) {
        unsigned long num_visits = position_count[probe_position];
        food_at_site[k] = food(probe_position);
        accumulated += exp(-food(probe_position)*num_visits);
      } else {
        // the counting_num_visits walk only makes sense as a myopic walk!?
        //food_at_site[k] = 0.0;
        //accumulated += 1.0;
      }
      #endif
      current_accprob[k] = accumulated;
    }
    #ifdef TEST_MULTI_EAT
      bool have_eaten = false;
    #endif
    for (unsigned int k=0; k<dim*2; ++k) {
      if (r < current_accprob[k]/accumulated) {
        newpos = walkpos + direction[k];
        accept_move = allowed[k];
        if (accept_move) {
          walkpos = newpos;
          eaten += food_at_site[k];
          #ifdef TEST_MULTI_EAT
            if (food_at_site[k]>0) have_eaten = true;
          #endif
        }
        break;
      }
    }
#if 0
    position_list[i] = walkpos;
#endif
    #ifdef PRINT_WALK
    // to print the walk:
    std::cout << i << "\t" << walkpos << std::endl;
    #endif
    //
    // add the new position to the position_hash
    // the hash's insert method returns an (iterator,boolean) pair,
    // where the boolean is true if the insertion succeeded (i.e. we
    // have a encountered new position)
    bool is_new_position = position_hash.insert(walkpos).second;
    #ifdef TEST_MULTI_EAT
      // (tv) 2018-03-07: suggestion by Peter Grassberger, 2018-02-23:
      // make the walker eat food also from adjacent sites
      // if we do this for all neighboring sites always, we will fall
      // back to the ordinary random walk
      // thus, only do this if we have actually eaten something in the
      // current step
      // we can (should) adjust the average percentage of n.n. food to be eaten;
      // looks like if we set that to 1, the exponent goes back to normal rw
      if (have_eaten) {
        eaten_hash.insert(walkpos);
        for (unsigned int k=0; k<dim*2; ++k) {
          double r = random_number();
          if (r<0.5) { // currently: eat half of the n.n. food
            pos_t probe_position = walkpos + direction[k];
            allowed[k] = matrix.allowed(probe_position);
            if (allowed[k]) {
              if (eaten_hash.find(probe_position) == eaten_hash.end())
                eaten += food(probe_position);
              eaten_hash.insert(probe_position);
            }
          }
        }
      }
    #endif
    // we keep track of the number of distinct sites this walk has visited
    if (is_new_position) num_distinct_sites++;
    #ifdef BIAS_NUM_VISITS
      position_count[walkpos] = position_count[walkpos] + 1;
    #endif
    // we can optionally also keep track of the fraction of walks
    // that stay on the backbone
    // (if the matrix generator supports it)
    if (count_backbone) {
      if (matrix.on_backbone(walkpos)) ++backbone_count;
    }
    //
    if (i>=printi) {
      // Calculation of observables (only done sparsely later in the walk).
      res.times[ti] = double(i);
      // All the accumulated values will later be divided by Nruns.
      // We calculate the "transient" MSD, i.e., relative to time zero and
      // the starting position; we do not average over several initial points
      // along the time series. Recall that a food-consuming walker is not
      // in a stationary state...
      double msdi = 0.0;
      for (unsigned int d=0; d<dim; ++d)
        msdi += pow<2>(double(walkpos[d]-startpos[d]));
      res.msd[ti] += msdi;
      // average of the number of sites visited
      res.distinct_sites[ti] += num_distinct_sites;
      // average of the number of non-visited neighbors just before this step
      res.filled_neighbors[ti] += filled;
      // amount of food eaten up to now
      res.food_consumed[ti] += eaten;
      // // average of the number of available neighbors just before this step
      // res.available_neighbors[ti] += available;
      #if COUNT_BACKBONE
      // walks that stay on the backbone
      res.backbone_count[ti] += backbone_count;
      #endif
      // set counter reflecting the next step for which observables shall
      // be calculated; if a block of length B is full, the stepping is doubled
      printi += iinc;
      if (++blocki>=B) { iinc*=2; blocki=B/2; }
      ++ti;
    }
    // calculation of van Hove functions
    // the compiler should optimize this away if calc_vanhove == false
    if (calc_vanhove) if (next_vh!=res.vanhove.end() && i == next_vh->first) {
      // we assume(!) the system to remain isotropic regarding rotations
      // hence, we calculate the van Hove function as a function of dx_i,
      // averaging over the spatial directions
      for (unsigned int d=0; d<dim; ++d) {
        double dr= walkpos[d] - startpos[d];
        if (dr<double(vanhove_cutoff) && dr>=-double(vanhove_cutoff)) {
          long drbin (dr);
          (next_vh->second)[drbin+vanhove_cutoff]++;
        }
      }
      #ifdef VH_LOG_MESSAGES
        std::cerr << "  reached t=" << i << std::endl;
      #endif
      ++next_vh;
    }
    #ifdef TIMED_LOG
      auto now = std::chrono::system_clock::now();
      if (std::chrono::duration_cast<std::chrono::milliseconds>
          (now - lastlogtime).count() > 1000000) {
        std::cerr << "  reached t = " << i << std::endl;
        lastlogtime = now;
      }
    #endif
  }

  return ti;
}

/** output **/

// print a file header summarizing the main parameters
template <typename food_class, typename engine>
std::ostream &header (std::ostream &os,
  const food_declaration<food_class> &food,
  const percolating_matrix<engine> &matrix) {
  #ifdef HGVERSION
    os << "# code version (hg): " HGVERSION "\n";
  #else
    os << "# unversioned code\n";
  #endif
  os << "# rng seed = " << rng_seed << "\n";
  food.info(os);
  os << "# dim = " << dim << "\tNruns = " << Nruns
     << "\tlength = " << length << "\n";
  matrix.info(os);
  return os;
}
// print averages stored in res
void write_averages (std::ostream &os, const results &res, unsigned long N,
  unsigned long imax) {
  unsigned long NN = N;
  #ifdef NO_AVERAGES
  NN = 1;
  #endif
  os <<
    "timestep\tMSD     \tdistinct_sites\tfilled_neighb\tfood_consumed"
    #if COUNT_BACKBONE
    "\ton_backbone"
    #endif
    "\n";
  for (unsigned long i=0; i<imax; ++i) {
    os << res.times[i]
       << "\t" << res.msd[i]/NN
       << "\t" << double(res.distinct_sites[i])/NN
       << "\t" << double(res.filled_neighbors[i])/NN
       << "\t" << res.food_consumed[i]/NN
       // << "\t" << double(res.available_neighbors[i])/NN
       #if COUNT_BACKBONE
       << "\t" << res.backbone_count[i]/NN
       #endif
       << std::endl;
  }
}

/** main **/

int main (int argc, char *argv []) {
  std::cerr << "reslength " << reslength << std::endl;
  parse_cmdline (argc, argv);
  initialize();

  auto food = FOOD;
  MATRIX matrix;
  try {
    matrix.init ();
  } catch (std::bad_alloc &) {
    std::cerr << "ERROR: matrix dimensions too large." << std::endl;
    return -9;
  }
  header (std::cout, food, matrix);
  header (std::cerr, food, matrix); // for the logfile
  unsigned long imax = 0;
  for (size_t m=0; m<Nmatrices; ++m) {
    matrix.realize (p);
    #pragma omp parallel
    {
    #if defined(_OPENMP) || defined(NO_AVERAGES)
    results locals;
    #else
    results &locals (globals);
    #endif
    #pragma omp for schedule(static)
    for (size_t n=0; n<Nruns; ++n) {
      std::cerr << "run " << n << " (matrix " << m << ")" << std::endl;
      imax = run (food,matrix,m*Nruns+n,locals);
      std::cerr << "done (run " << n << ")" << std::endl;
    }
    #pragma omp critical
    #if defined(_OPENMP) || defined(NO_AVERAGES)
    globals.add(locals);
    #endif
    }
    if (intermediate_averages && m+1<Nmatrices) {
      std::ostringstream filename; filename << "avg" << m+1;
      std::ofstream avgout (filename.str());
      avgout.setf(std::ios::scientific);
      header (avgout, food, matrix);
      avgout << "# INTERMEDIATE AVERAGES after " << m+1 << " matrices\n";
      write_averages (avgout, globals, (m+1)*Nruns, imax);
    }
  }
  #ifndef PRINT_WALK
  write_averages (std::cout, globals, Nruns*Nmatrices, imax);
  #endif
  std::cerr << "imax = " << imax << ", t = " << globals.times[imax-1]
            << std::endl;
  if (calc_vanhove) {
    unsigned long Ntotal = Nruns*Nmatrices;
    std::ofstream vhout ("vanhove.dat");
    header (vhout, food, matrix);
    vhout << "# vH(dx), one time per column:\n";
    vhout << "dx";
    for (auto t : globals.vanhove) vhout << "\t" << t.first;
    vhout << std::endl;
    //
    for (unsigned int x=0; x<2*vanhove_cutoff+1; ++x) {
      vhout << int(x)-int(vanhove_cutoff);
      for (auto t : globals.vanhove)
        vhout << "\t" << double((t.second)[x])/Ntotal/dim;
      vhout << std::endl;
    }
  }
  return 0;
}
