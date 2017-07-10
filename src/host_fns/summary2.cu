#include "../header/summary2.h"


struct is_zero: thrust::unary_function<int, bool>{
  __host__ __device__ bool operator()(int i){
    return i==0 ? 1 : 0;
  }
};

struct fi_multiply: thrust::binary_function<double, int, double>{
  __host__ __device__ double operator()(double x, int y){
    return x * y;
  }
};

// zeta passed by value, data passed by reference
summary2::summary2(int _K, ivec_d zeta, data_t &data): G(data.G), K(_K), V(data.V), occupied(_K), Mk(_K, 0){

  // allocate,initialize perm for storing zeta-inverse
  ivec_d perm(G); // will store permutation
  thrust::sequence(perm.begin(), perm.end(), 0, 1);

  // sort zeta, saving perm which is zeta-inverse
  // note: zeta is used as key for reduce_by_key
  thrust::sort_by_key(zeta.begin(), zeta.end(), perm.begin());

  //save occupied indices and num_occupied
  ivec_d::iterator endptr;
  endptr = thrust::unique_copy(zeta.begin(), zeta.end(), occupied.begin());
  occupied.erase(endptr, occupied.end());
  num_occupied = occupied.size();

  // calculate and save Mk, i.e. table(zeta)
  thrust::permutation_iterator<intIter, intIter> mapMk = thrust::permutation_iterator<intIter,intIter>(Mk.begin(), occupied.begin());
  thrust::reduce_by_key(zeta.begin(), zeta.end(), thrust::make_constant_iterator<int>(1), thrust::make_discard_iterator(), mapMk);

  // save unoccupied indices
  unoccupied.reserve(K - num_occupied);
  endptr = thrust::copy_if(thrust::make_counting_iterator<int>(0),
                           thrust::make_counting_iterator<int>(K), Mk.begin(), unoccupied.begin(), is_zero());
  unoccupied.erase(endptr, unoccupied.end());

  //resize vectors for reductions
  yty_sums.resize(num_occupied);
  xty_sums.resize(num_occupied*V);
  ytx_sums.resize(num_occupied*V);
  size_t sz_xtx = num_occupied * V * V;
  xtx_sums.resize(sz_xtx);
  
  /*yty_sums
   *
   */
  // permutation_iterator for reduce_by_key
  thrust::permutation_iterator<realIter, intIter> sort_yty = thrust::permutation_iterator<realIter, intIter>(data.yty.begin(), perm.begin());
  thrust::reduce_by_key(zeta.begin(), zeta.end(), sort_yty, thrust::make_discard_iterator(), yty_sums.begin());

  /* ytx_sums
   * 
   */
  // repeat zeta key, to reduce by key each element of ytx (repeated-shifted-iterator)
  RSIntIter zeta_rep = getRSIntIter(zeta.begin(), zeta.end(), K);
  // permutation_iterator for reduce by key (multiple elements)
  RSIntIter in_index = getRSIntIter(perm.begin(),perm.end(), G);
  thrust::permutation_iterator<realIter, RSIntIter> sort_ytx = thrust::permutation_iterator<realIter, RSIntIter>(data.ytx.begin(), in_index);
  thrust::reduce_by_key(zeta_rep, zeta_rep + G*V, sort_ytx, thrust::make_discard_iterator(), ytx_sums.begin());

  /* xty_sums
  * 
    */
    transpose<realIter>(ytx_sums.begin(), ytx_sums.end(), num_occupied, V, xty_sums.begin());

  /* xtx_sums
  *
  */
  voom = data.voom; // whether or not W_g depends on g
  
  if(!voom){
    //copy xtx over (repeating loop)
    realIter xtx_begin = data.xtx.begin();
    realIter xtx_end = data.xtx.end();
    gRepTimes<realIter>::iterator xtx_rep = getGRepTimesIter(xtx_begin, xtx_end, V*V, 1); 
    thrust::copy(xtx_rep, xtx_rep + sz_xtx, xtx_sums.begin());
    
    //multiply xtx by occupied Mk (table(zeta))
    thrust::permutation_iterator<intIter, intIter> Mk_occ = thrust::permutation_iterator<intIter, intIter>(Mk.begin(), occupied.begin());
    gRepEach<thrust::permutation_iterator<intIter,intIter> >::iterator Mk_rep = getGRepEachIter(Mk_occ, Mk_occ + num_occupied, V*V, 1);
    transform(xtx_sums.begin(), xtx_sums.begin() + sz_xtx, Mk_rep, xtx_sums.begin(), thrust::placeholders::_1 * thrust::placeholders::_2);
    
  } else{
    // temporary used for reduction only
    fvec_d txtx_sums(sz_xtx);
    
    // reusing permutation iterator from ytx_sums, make permutation_iterator for reduce_by_key  
    thrust::permutation_iterator<realIter, RSIntIter> sort_txtx = thrust::permutation_iterator<realIter, RSIntIter>(data.txtx.begin(), in_index);
    thrust::reduce_by_key(zeta_rep, zeta_rep + G*V*V, sort_txtx, thrust::make_discard_iterator(), txtx_sums.begin());
    
    //transpose into xtx_sums which is a member of summary2
    transpose<realIter>(txtx_sums.begin(), txtx_sums.end(), num_occupied, V*V, xtx_sums.begin());
  }
}

typedef thrust::tuple<realIter,realIter,realIter> tup3;
typedef thrust::zip_iterator<tup3> zip3;

void summary2::sumSqErr(fvec_d &sse, fvec_d &beta, int verbose=0){

  // strip unoccupied betas for calculations
  fvec_d beta_occ(num_occupied*V);
  
  // indices for occupied betas
  SCIntIter beta_id = getSCIntIter(occupied.begin(), occupied.end(), V);
  thrust::copy(thrust::make_permutation_iterator(beta.begin(), beta_id),
               thrust::make_permutation_iterator(beta.begin(), beta_id) + num_occupied*V,
               beta_occ);


  /***
  * calculate t(beta)*xtx_k*beta --> sse
  ***/
  
  quadform_multipleMatch(beta_occ, xtx_sums, sse, num_occupied, V);

  // Print value
  if(verbose>0){
    std::cout << "\nbxxb\n";
    printVec(sse, num_occupied, 1);
  }
  
  /***
  * calculate -2 * ytx_k * beta
  ***/
  
  fvec_d ytxb(num_occupied);
  multi_dot_prod(beta_occ, xty_sums, ytxb, V, num_occupied);
  if(verbose>0){
    std::cout << "\nytxb:\n";
    printVec(ytxb, num_occupied, 1);
  }
  thrust::transform(ytxb.begin(), ytxb.end(), ytxb.begin(), -2.0 * thrust::placeholders::_1);
  if(verbose>0){
    std::cout << "\nytxb * -2:\n";
    printVec(ytxb, num_occupied, 1);
  }
  
  /***
  * add three numbers together
  ***/
  tup3 my_tuple = thrust::make_tuple(sse.begin(), ytxb.begin(), yty_sums.begin());
  zip3 my_zip = thrust::make_zip_iterator(my_tuple);
  thrust::for_each(my_zip, my_zip + num_occupied, add3());
  if(verbose>0){
    std::cout << "yty_sums:\n";
    printVec(yty_sums, num_occupied, 1);
    std::cout << "3 added:\n";
    printVec(sse, num_occupied, 1);
  }
}
