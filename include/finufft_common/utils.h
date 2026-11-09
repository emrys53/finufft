#pragma once

#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "constants.h"
#include "defines.h"

// Runtime->compile-time dispatch (formerly make_range/DispatchParam/dispatch,
// defined here) now comes from POET, included only in the TUs that dispatch:
//   make_range    -> poet::inclusive_range
//   DispatchParam -> poet::dispatch_param
//   dispatch      -> poet::dispatch

namespace finufft {
namespace common {

// Returns p*required factor, where p is the smallest composite
// of 2, 3, 5 such that p*required_factor >= n.
BIGINT next235(BIGINT n, BIGINT required_factor = 1);

FINUFFT_EXPORT_TEST void gaussquad(int n, double *xgl, double *wgl);
std::tuple<double, double> leg_eval(int n, double x);

} // namespace common
} // namespace finufft

namespace finufft::utils {

// Round v down / up to a multiple of align. Every align used here is a SIMD
// width or a kernel-buffer stride, i.e. a power of two; that is the property
// the mask form relies on, so it is checked rather than assumed. The runtime
// overloads exist so that a compile-time formula and its runtime mirror cannot
// drift apart (see kernel_buffer_stride_runtime in simd.hpp).
template<typename T> constexpr T round_down(T v, std::size_t align) noexcept {
  assert(align != 0 && (align & (align - 1)) == 0); // power of two
  return v & ~T(align - 1);
}
template<typename T> constexpr T round_up(T v, std::size_t align) noexcept {
  return round_down(T(v + T(align - 1)), align);
}
template<std::size_t Align, typename T> constexpr T round_down(T v) noexcept {
  static_assert(Align != 0 && (Align & (Align - 1)) == 0, "Align must be a power of two");
  static_assert(Align - 1 <= std::size_t((std::numeric_limits<T>::max)()),
                "Align does not fit in T; the mask would wrap");
  return round_down(v, Align);
}
template<std::size_t Align, typename T> constexpr T round_up(T v) noexcept {
  return round_down<Align>(T(v + T(Align - 1)));
}

// Host versions of arrayrange / arraywidcen. The CUDA path has a separate
// device-pointer overload (in include/cufinufft/utils.hpp) that uses thrust.
template<typename T>
FINUFFT_ALWAYS_INLINE void arrayrange(int64_t n, const T *a, T *lo, T *hi)
// With a a length-n array, writes out min(a) to lo and max(a) to hi,
// so that all a values lie in [lo,hi].
// If n==0, lo and hi are not finite.
{
  *lo = INFINITY;
  *hi = -INFINITY;
  for (int64_t m = 0; m < n; ++m) {
    if (a[m] < *lo) *lo = a[m];
    if (a[m] > *hi) *hi = a[m];
  }
}
template<typename T>
FINUFFT_ALWAYS_INLINE void arraywidcen(int64_t n, const T *a, T *w, T *c)
// Writes out w = half-width and c = center of an interval enclosing all a[n]'s
// Only chooses a nonzero center if this increases w by less than fraction
// ARRAYWIDCEN_GROWFRAC defined in finufft_common/constants.h.
// This prevents rephasings which don't grow nf by much. 6/8/17
// If n==0, w and c are not finite.
{
  T lo, hi;
  arrayrange(n, a, &lo, &hi);
  *w = (hi - lo) / 2;
  *c = (hi + lo) / 2;
  if (std::abs(*c) < finufft::common::ARRAYWIDCEN_GROWFRAC * (*w)) {
    *w += std::abs(*c);
    *c = 0.0;
  }
}

// Like std::allocator, but vector(n) and resize(n) leave new elements uninitialised.
template<typename T, std::size_t Align = alignof(T)>
struct uninitialized_allocator {
  static_assert(Align >= alignof(T) && (Align & (Align - 1)) == 0,
                "Align must be a power of two and at least alignof(T)");
  using value_type      = T;
  using is_always_equal = std::true_type;

  // keep no-init and Align when std::vector rebinds
  template<typename U>
  struct rebind {
    using other = uninitialized_allocator<U, (Align > alignof(U) ? Align : alignof(U))>;
  };

  uninitialized_allocator() = default;
  template<typename U, std::size_t A>
  uninitialized_allocator(const uninitialized_allocator<U, A> &) noexcept {}

  [[nodiscard]] T *allocate(std::size_t n) {
    return static_cast<T *>(::operator new(n * sizeof(T), std::align_val_t{Align}));
  }
  void deallocate(T *p, std::size_t) noexcept {
    ::operator delete(p, std::align_val_t{Align});
  }

  template<typename U>
  void construct(U *) noexcept {
    static_assert(std::is_trivially_copyable_v<U> && std::is_trivially_destructible_v<U>,
                  "uninitialized_allocator only works with trivially copyable and "
                  "trivially destructible types");
  }
  template<typename U, typename... Args>
  void construct(U *p, Args &&...args) {
    ::new (static_cast<void *>(p)) U(std::forward<Args>(args)...);
  }

  template<typename U, std::size_t A>
  bool operator==(const uninitialized_allocator<U, A> &) const noexcept {
    return A == Align;
  }
  template<typename U, std::size_t A>
  bool operator!=(const uninitialized_allocator<U, A> &) const noexcept {
    return A != Align;
  }
};
template<typename T, std::size_t Align = alignof(T)>
using uninitialized_vector = std::vector<T, uninitialized_allocator<T, Align>>;

} // namespace finufft::utils
