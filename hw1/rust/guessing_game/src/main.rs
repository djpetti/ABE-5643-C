use rand::prelude::*;
use std::io::{stdin, stdout, Write};

/// Reads an integer from the terminal.
fn read_int() -> u32 {
    let mut guess_str = String::new();
    stdin()
        .read_line(&mut guess_str)
        .expect("Please enter a number.");
    let guess_trimmed = guess_str.trim();
    guess_trimmed.parse::<u32>().unwrap()
}

fn main() {
    let answer = random::<u32>() % 100 + 1;

    let mut guess: u32 = 0;
    while guess != answer {
        print!("Guess a number from 1 to 100: ");
        let _ = stdout().flush();
        guess = read_int();

        if guess > answer {
            println!("Too high, guess again.");
        } else if guess < answer {
            println!("Too low, guess again.");
        } else {
            println!("You guessed the number! You're a genius!");
        }
    }
}
