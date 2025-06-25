use libc::{c_int, c_uint, getrandom};

use crate::{num::traits::BinaryInteger, traits::RandomNumberGenerator};

/// A system-based cryptographically secure random number generator.
///
/// `SystemRandomNumberGenerator` utilizes operating system facilities to generate
/// high-quality, cryptographically secure random bytes. It abstracts over the
/// underlying platform details while providing a consistent API.
///
/// ## Platform Behavior
///
/// | Platform | Source               | Status       |
/// |----------|----------------------|--------------|
/// | **macOS**    | `arc4random_buf`       | ✅ Supported |
/// | **Windows**  | `BCryptGenRandom`      | ✅ Supported |
/// | **Linux**    | `getrandom` or `/dev/urandom` | ✅ Supported |
/// | **Others**   | *(none)*               | ❌ Unsupported (panics) |
///
/// ## Panics
///
/// This method will panic in the following cases:
///
/// - On **Windows**, if `BCryptGenRandom` fails.
/// - On **Linux**, if both `getrandom` and `/dev/urandom` are unavailable.
/// - On **unsupported platforms**, where no secure RNG is implemented.
///
/// ## Examples
///
/// ```
/// use foundation::random::SystemRandomNumberGenerator;
///
/// let rng = SystemRandomNumberGenerator::new();
/// let mut buffer = [0u8; 32];
/// rng.fill_bytes(&mut buffer);
/// println!("Random bytes: {:?}", buffer);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct SystemRandomNumberGenerator;

impl SystemRandomNumberGenerator {
    /// Creates a new `SystemRandomNumberGenerator`.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Fills the given buffer with cryptographically secure random bytes using the system RNG.
    ///
    /// - On **macOS**, uses `arc4random_buf`.
    /// - On **Windows**, uses `BCryptGenRandom`.
    /// - On **Linux**, attempts `getrandom` syscall, falls back to `/dev/urandom` if necessary.
    /// - On other platforms, this will panic.
    ///
    /// ## Parameters
    ///
    /// - `buf`: A mutable slice that will be filled with random data.
    ///
    /// ## Panics
    ///
    /// - On Linux, if both `getrandom` and `/dev/urandom` are unavailable or fail.
    /// - On Windows, if `BCryptGenRandom` returns a failure status.
    /// - On unsupported platforms.
    ///
    /// ## Examples
    ///
    /// ```
    /// use foundation::random::SystemRandomNumberGenerator;
    ///
    /// let rng = SystemRandomNumberGenerator::new();
    /// let mut data = [0u8; 16];
    /// rng.fill_bytes(&mut data);
    /// ```
    pub fn fill_bytes(&self, buf: &mut [u8]) {
        cfg_select! {
            target_os = "linux" => {
                const GRND_NONBLOCK: c_uint = 0x0001;
                const ENOSYS: c_int = 38;

                let mut filled = 0;
                while filled < buf.len() {
                    let result = unsafe { getrandom(buf[filled..].as_mut_ptr().cast(), buf.len() - filled, GRND_NONBLOCK) };
                    if result < 0 {
                        let err = unsafe { *libc::__errno_location() };
                        if err == ENOSYS {
                            let fd = unsafe { libc::open(c"/dev/urandom".as_ptr(), libc::O_RDONLY) };
                            if fd < 0 {
                                panic!("Failed to open /dev/urandom");
                            }
                            let read_result = unsafe { libc::read(fd, buf[filled..].as_mut_ptr().cast(), buf.len() - filled) };
                            if read_result < 0 {
                                panic!("Failed to read from /dev/urandom");
                            }
                            filled += read_result as usize;
                            unsafe { libc::close(fd) };
                        } else {
                            panic!("getrandom failed with error: {}", err);
                        }
                    } else {
                        filled += result as usize;
                    }
                }
            }

            target_os = "macos" => {
                unsafe {
                    use libc::arc4random_buf;
                    arc4random_buf(buf.as_mut_ptr().cast(), buf.len());
                }
            }

            target_os = "windows" => {
                unsafe {
                    use core::ptr::null_mut;
                    use windows_sys::Win32::Security::Cryptography::{
                        BCRYPT_USE_SYSTEM_PREFERRED_RNG, BCryptGenRandom,
                    };

                    let status = BCryptGenRandom(
                        null_mut(),
                        buf.as_mut_ptr(),
                        buf.len() as u32,
                        BCRYPT_USE_SYSTEM_PREFERRED_RNG,
                    );

                    assert!(
                        (status == 0),
                        "BCryptGenRandom failed with status: 0x{status:X}"
                    );
                }
            }

            _ => {
                panic!("SystemRandomNumberGenerator is not supported on this platform. Please implement a custom RNG or use a different one.");
            }
        }
    }
}

impl Default for SystemRandomNumberGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl RandomNumberGenerator for SystemRandomNumberGenerator {
    fn next<T: Sized + Copy + Default>(&mut self) -> T {
        let mut val: T = Default::default();
        let ptr = (&raw mut val).cast::<u8>();
        let buf = unsafe { alloc::slice::from_raw_parts_mut(ptr, size_of::<T>()) };
        self.fill_bytes(buf);
        val
    }

    fn next_below<T>(&mut self, upper_bound: T) -> T
    where
        T: BinaryInteger + Default,
    {
        let value: T = self.next();
        value % upper_bound
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_next_u32_returns_different_values() {
        let mut rng = SystemRandomNumberGenerator::new();
        let a: u32 = rng.next();
        let b: u32 = rng.next();
        // It's possible they match, but very unlikely!
        assert_ne!(
            a, b,
            "Two successive values should not be equal most of the time"
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_next_below_respects_upper_bound() {
        let mut rng = SystemRandomNumberGenerator::new();
        for _ in 0..100 {
            let val: u32 = rng.next_below(100);
            assert!(val < 100, "Value {val} is not less than 100");
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_fill_bytes_fills_buffer() {
        let rng = SystemRandomNumberGenerator::new();
        let mut buffer = [0u8; 32];
        rng.fill_bytes(&mut buffer);
        assert!(
            buffer.iter().any(|&b| b != 0),
            "Buffer contains only zeroes"
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_randomness_stays_within_bounds_for_various_types() {
        let mut rng = SystemRandomNumberGenerator::new();
        let max = 250u8;
        for _ in 0..100 {
            let x: u8 = rng.next_below(max);
            assert!(x < max);
        }

        let max = 10_000u16;
        for _ in 0..100 {
            let x: u16 = rng.next_below(max);
            assert!(x < max);
        }
    }
}
