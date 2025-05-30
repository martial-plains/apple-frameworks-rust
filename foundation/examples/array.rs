use foundation::{array, collections::Sequence};

fn main() {
    let s = array![1, 2, 3];
    for (n, x) in s.clone().enumerated() {
        println!("{n}: {x} = {}", s[n]);
    }
}
