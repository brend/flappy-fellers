use macroquad::{
    color::*,
    input::{KeyCode, is_key_pressed},
    prelude::ImageFormat,
    shapes::draw_rectangle,
    text::draw_text,
    texture::{Texture2D, draw_texture},
    window::{clear_background, next_frame, screen_height, screen_width},
};
use neural_network_study::{ActivationFunction, NeuralNetwork};
use rand::prelude::*;
use serde::{Deserialize, Serialize};

/// Speed at which the pipes move in pixels per iteration
const HSPEED: f32 = 0.8;
/// Maximum vertical speed of a flappy feller
const FELLER_MAX_SPEED: f32 = 2.0;
/// Probability of spawning a pipe during an iteration
const PIPE_PROBABILITY: f32 = 0.002;
/// Pipe width
const PIPE_WIDTH: f32 = 40.0;
/// Minimum size of the pipe aperture (hole)
const PIPE_MIN_APERTURE: f32 = 80.0;
/// Maximum size of the pipe aperture
const PIPE_MAX_APERTURE: f32 = 160.0;
/// Minimum distance between two pipes
const PIPE_MIN_DISTANCE: f32 = 160.0;
/// Jumping force
const LIFT: f32 = 2.0;
/// x-coordinate of the fellers
const FELLER_X: f32 = 40.0;
/// body radius of the fellers
const FELLER_R: f32 = 20.0;
/// Number of fellers in each generation
const POPULATION_SIZE: usize = 150;
/// Probability of mutation of weights
/// during cloning of neural network
const MUTATION_RATE: f64 = 0.1;

/// main function simulates and displays the game
#[macroquad::main("Flappy Feller")]
async fn main() {
    let mut rng = StdRng::from_os_rng();
    // a vec to hold the pipes for fellers to crash into
    let mut pipes: Vec<Pipe> = vec![];
    // a collection of fellers flapping alongside each other
    let mut population = Population::new(POPULATION_SIZE);
    // the number of steps simulated during each frame.
    // this allows to speed up the training process
    let mut iterations_per_frame = 1;
    // step counter used to score the fellers
    let mut steps = 0;
    // generation counter used purely for visualization
    let mut generation = 1;
    // graphics resources
    let walden_sprite = Texture2D::from_file_with_format(
        include_bytes!("../assets/walden.png"),
        Some(ImageFormat::Png),
    );

    loop {
        handle_input(&mut iterations_per_frame);

        // simulate one or more steps of the game
        for _ in 0..iterations_per_frame {
            simulate_step(&mut pipes, &mut population.fellers, &mut rng, steps);
            steps += 1;
        }

        // spawn a new population once the current one has expired
        if !population.is_alive() {
            steps = 0;
            generation += 1;
            pipes.clear();
            population = Population::from_predecessors(population);
        }

        draw_scene(
            &pipes,
            &population,
            &walden_sprite,
            generation,
            iterations_per_frame,
        );

        next_frame().await
    }
}

/// Draws the game state: fellers, pipes and HUD
fn draw_scene(
    pipes: &[Pipe],
    population: &Population,
    walden_sprite: &Texture2D,
    generation: usize,
    iterations_per_frame: usize,
) {
    // draw the scene
    clear_background(WHITE);

    // draw pipes
    for pipe in pipes {
        draw_rectangle(pipe.x, 0.0, PIPE_WIDTH, pipe.y1, BLACK);
        draw_rectangle(
            pipe.x,
            pipe.y2,
            PIPE_WIDTH,
            screen_height() - pipe.y2,
            BLACK,
        );
    }

    // draw the feller
    for feller in &population.fellers {
        if feller.is_alive {
            // let color = Color::from_rgba(0, 0, 0, 64);
            // draw_circle(FELLER_X, feller.y, FELLER_R, color);
            draw_texture(
                walden_sprite,
                FELLER_X,
                feller.y,
                Color::from_rgba(255, 255, 255, 100),
            );
        }
    }

    // draw the HUD
    draw_text(
        &format!(
            "Generation {}; Fellers: {}; Speed: {}",
            generation,
            population.survivor_count(),
            iterations_per_frame
        ),
        20.0,
        20.0,
        20.0,
        BLUE,
    );
}

/// Handle keyboard input from the user
fn handle_input(iterations_per_frame: &mut usize) {
    // speeding up and slowing down the simulation
    if is_key_pressed(KeyCode::S) {
        *iterations_per_frame += 1;
    } else if is_key_pressed(KeyCode::A) {
        *iterations_per_frame -= 1;
    } else if is_key_pressed(KeyCode::Key0) {
        *iterations_per_frame = 1;
    } else if is_key_pressed(KeyCode::Key9) {
        *iterations_per_frame += 10;
    }

    *iterations_per_frame = (*iterations_per_frame).clamp(1, 100);
}

/// Simulates a single step of the game
fn simulate_step(pipes: &mut Vec<Pipe>, fellers: &mut [Feller], rng: &mut StdRng, step: i32) {
    simulate_pipes(pipes, rng);

    for feller in fellers.iter_mut() {
        if feller.is_alive {
            simulate_feller(feller, pipes, step);
        }
    }
}

/// Move the pipes ahead, occasionally spawning new ones
fn simulate_pipes(pipes: &mut Vec<Pipe>, rng: &mut StdRng) {
    // spawn a new pipe with a certain probability
    if pipes.is_empty() || rng.random::<f32>() < PIPE_PROBABILITY {
        let spawn_allowed = match pipes.last() {
            Some(pipe) => pipe.x + PIPE_MIN_DISTANCE < screen_width(),
            None => true,
        };
        if spawn_allowed {
            pipes.push(Pipe::random(rng));
        }
    }

    // update pipes
    for pipe in pipes.iter_mut() {
        pipe.x -= HSPEED;
    }

    // remove pipes that have left the screen
    pipes.retain(|p| p.x + PIPE_WIDTH > 0.0);
}

/// Move a feller according to gravity and input (jumping)
/// and check for collisions with environment objects
fn simulate_feller(feller: &mut Feller, pipes: &mut Vec<Pipe>, step: i32) {
    // update the feller based on the neural network's output
    let closest_pipe = pipes.iter().find(|&p| p.x > FELLER_X);
    if let Some(pipe) = closest_pipe {
        let w = screen_width();
        let h = screen_height();
        let input = vec![
            (feller.y / h) as f64,
            (feller.yspeed / FELLER_MAX_SPEED) as f64,
            (pipe.x / w) as f64,
            (pipe.y1 / h) as f64,
            (pipe.y2 / h) as f64,
        ];
        let output = feller.predict(input);
        if output[0] > output[1] {
            feller.yspeed -= LIFT;
        }
    }

    // Update the feller's vertical speed with gravitation
    feller.yspeed = (feller.yspeed + 0.02).clamp(-FELLER_MAX_SPEED, FELLER_MAX_SPEED);
    feller.y += feller.yspeed;

    // Check for collisions with ceiling and floor
    if feller.y < 0.0 || feller.y > screen_height() {
        feller.is_alive = false;
        feller.steps_survived = step
    }

    // Check for collisions with pipes
    for pipe in pipes {
        if (pipe.x - FELLER_X).abs() < FELLER_R
            && (feller.y - FELLER_R < pipe.y1 || feller.y + FELLER_R > pipe.y2)
        {
            feller.is_alive = false;
            feller.steps_survived = step
        }
    }
}

/// Compute a score for a feller
fn score(feller: &Feller) -> f32 {
    feller.steps_survived as f32
}

/// Pipes are the fellers' main obstacles.
/// Fellers must fly through the hole in the middle
/// of the pipe to survive.
struct Pipe {
    /// x-coordinate of the pipe
    x: f32,
    /// y-coordinate of the top of the hole
    y1: f32,
    /// y-coordinate of the bottom of the hole
    y2: f32,
}

impl Pipe {
    /// create a pipe with a randomized hole
    pub fn random(rng: &mut StdRng) -> Pipe {
        let y1 = rng.random_range(100.0..200.0);
        let y2 = y1 + rng.random_range(PIPE_MIN_APERTURE..PIPE_MAX_APERTURE);
        Pipe {
            x: screen_width(),
            y1,
            y2,
        }
    }
}

/// Flappy fellers are the heroes of this story.
/// They are being controlled by an evolving AI.
#[derive(Serialize, Deserialize)]
struct Feller {
    /// y-coordinate of the feller
    y: f32,
    /// vertical speed of the feller
    yspeed: f32,
    /// the AI that controls the feller
    brain: NeuralNetwork,
    /// flag indicating if the feller is still alive
    is_alive: bool,
    /// counter of how many simulation steps
    /// this feller has survived
    steps_survived: i32,
}

impl Feller {
    /// Creates a new feller with a randomized neural network
    /// for a brain and at the default height
    fn new() -> Feller {
        let mut rng = StdRng::from_os_rng();
        let mut brain = NeuralNetwork::new(5, 4, 2, Some(&mut rng));
        brain.set_activation_function(ActivationFunction::Sigmoid);
        Feller {
            y: screen_height() / 3.0,
            yspeed: 0.0,
            brain,
            is_alive: true,
            steps_survived: 0,
        }
    }

    /// ask the feller for their move during a simulation step
    fn predict(&self, input: Vec<f64>) -> Vec<f64> {
        self.brain.predict(input)
    }

    /// mutate the brain of the feller 🧟‍♂️
    fn mutate(&mut self, rng: &mut StdRng) {
        self.brain.mutate(rng, MUTATION_RATE);
    }

    /// create a new feller with a clone of this one's brain
    /// and otherwise default properties
    fn spawn(&self) -> Feller {
        Feller {
            y: screen_height() / 3.0,
            yspeed: 0.0,
            brain: self.brain.clone(),
            is_alive: true,
            steps_survived: 0,
        }
    }
}

/// Population of fellers that competete against each other
struct Population {
    /// The fellers of this generation
    fellers: Vec<Feller>,
}

impl Population {
    /// Create a new population of fellers of the desired size
    fn new(size: usize) -> Population {
        let mut fellers = vec![];
        for _ in 0..size {
            fellers.push(Feller::new());
        }
        Population { fellers }
    }

    /// Determines if the population is still alive.
    /// A population is alive if at least one of its fellers
    /// is still alive.
    fn is_alive(&self) -> bool {
        for feller in &self.fellers {
            if feller.is_alive {
                return true;
            }
        }
        false
    }

    /// Returns the number of fellers that are still alive
    fn survivor_count(&self) -> usize {
        let mut count = 0;
        for feller in &self.fellers {
            if feller.is_alive {
                count += 1;
            }
        }
        count
    }

    /// Spawn a new population of fellers
    /// by scoring the ones in this generation
    /// and cloning the best ones
    fn from_predecessors(predecessors: Population) -> Population {
        // compute a score for each feller
        // then sort them by descending score
        // and retain only the top 5%
        let mut scored_fellers = predecessors
            .fellers
            .into_iter()
            .map(|p| (score(&p), p))
            .map(|(s, p)| (s * s, p))
            .collect::<Vec<_>>();
        scored_fellers.sort_by(|a, b| b.0.total_cmp(&a.0));
        let keep_len = (POPULATION_SIZE as f64 * 0.05).ceil() as usize;
        scored_fellers.truncate(keep_len);

        // normalize scores
        let mut score_sum = 0.0;
        for (score, _) in &scored_fellers {
            score_sum += score;
        }
        let scored_fellers = scored_fellers
            .into_iter()
            .map(|(s, f)| (s / score_sum, f))
            .collect::<Vec<_>>();

        let mut descendants = vec![];
        let mut rng = StdRng::from_os_rng();

        // create 80% of the new descendants by random picking
        // while the highest scorers are most likely to procreate
        let procreation_len = (0.8 * POPULATION_SIZE as f64).ceil() as usize;
        while descendants.len() < procreation_len {
            let mut r = rng.random_range(0.0..1.0);

            for (score, feller) in &scored_fellers {
                r -= score;
                if r <= 0.0 {
                    let mut child = feller.spawn();
                    child.mutate(&mut rng);
                    descendants.push(child);
                    break;
                }
            }
        }

        // fill up the remaining slots with random new fellers
        while descendants.len() < POPULATION_SIZE {
            descendants.push(Feller::new());
        }

        Population {
            fellers: descendants,
        }
    }
}
