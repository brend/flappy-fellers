use macroquad::{
    color::*,
    input::{KeyCode, is_key_pressed},
    shapes::{draw_circle, draw_rectangle},
    text::draw_text,
    window::{clear_background, next_frame, screen_height, screen_width},
};
use neural_network_study::{ActivationFunction, NeuralNetwork};
use rand::prelude::*;

const HSPEED: f32 = 0.8;
const MAX_SPEED: f32 = 2.0;

const PIPE_PROBABILITY: f32 = 0.002;
const PIPE_WIDTH: f32 = 40.0;
const PIPE_MIN_APERTURE: f32 = 80.0;
const PIPE_MAX_APERTURE: f32 = 160.0;
const PIPE_MIN_DISTANCE: f32 = 160.0;

const LIFT: f32 = 2.0;
const FELLER_X: f32 = 40.0;
const FELLER_R: f32 = 20.0;

const POPULATION_SIZE: usize = 150;
const MUTATION_RATE: f64 = 0.1;

#[macroquad::main("Flappy Feller")]
async fn main() {
    let mut rng = StdRng::from_os_rng();
    let mut pipes: Vec<Pipe> = vec![];
    let mut population = Population::new(POPULATION_SIZE);
    let mut iterations_per_frame = 1;
    let mut steps = 0;
    let mut generation = 1;

    loop {
        clear_background(WHITE);

        if is_key_pressed(KeyCode::S) {
            iterations_per_frame += 1;
        } else if is_key_pressed(KeyCode::A) {
            iterations_per_frame -= 1;
        } else if is_key_pressed(KeyCode::Key0) {
            iterations_per_frame = 1;
        } else if is_key_pressed(KeyCode::Key9) {
            iterations_per_frame += 10;
        }
        iterations_per_frame = iterations_per_frame.clamp(1, 100);

        for _ in 0..iterations_per_frame {
            simulate_step(&mut pipes, &mut population.fellers, &mut rng, steps);
            steps += 1;
        }

        // check if population has died out
        if !population.is_alive() {
            steps = 0;
            generation += 1;
            pipes.clear();
            population = Population::from_predecessors(population);
        }

        // draw pipes
        for pipe in &pipes {
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
                let color = Color::from_rgba(0, 0, 0, 64);
                draw_circle(FELLER_X, feller.y, FELLER_R, color);
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

        next_frame().await
    }
}

fn simulate_step(pipes: &mut Vec<Pipe>, fellers: &mut Vec<Feller>, rng: &mut StdRng, step: i32) {
    simulate_pipes(pipes, rng);

    for feller in fellers.iter_mut() {
        if feller.is_alive {
            simulate_feller(feller, pipes, step);
        }
    }
}

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
    pipes.retain(|p| p.x + PIPE_WIDTH > 0.0);
}

fn simulate_feller(feller: &mut Feller, pipes: &mut Vec<Pipe>, step: i32) {
    // update the feller based on the neural network's output
    let closest_pipe = pipes.iter().find(|&p| p.x > FELLER_X);
    if let Some(pipe) = closest_pipe {
        let w = screen_width();
        let h = screen_height();
        let input = vec![
            (feller.y / h) as f64,
            (feller.yspeed / MAX_SPEED) as f64,
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
    feller.yspeed = (feller.yspeed + 0.02).clamp(-MAX_SPEED, MAX_SPEED);
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

fn score(feller: &Feller) -> f32 {
    feller.steps_survived as f32
}

struct Pipe {
    x: f32,
    y1: f32,
    y2: f32,
}

impl Pipe {
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

struct Feller {
    y: f32,
    yspeed: f32,
    brain: NeuralNetwork,
    is_alive: bool,
    steps_survived: i32,
}

impl Feller {
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

    fn predict(&self, input: Vec<f64>) -> Vec<f64> {
        self.brain.predict(input)
    }

    fn mutate(&mut self, rng: &mut StdRng) {
        self.brain.mutate(rng, MUTATION_RATE);
    }

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

struct Population {
    fellers: Vec<Feller>,
}

impl Population {
    fn new(size: usize) -> Population {
        let mut fellers = vec![];
        for _ in 0..size {
            fellers.push(Feller::new());
        }
        Population { fellers }
    }

    fn is_alive(&self) -> bool {
        for feller in &self.fellers {
            if feller.is_alive {
                return true;
            }
        }
        return false;
    }

    fn survivor_count(&self) -> usize {
        let mut count = 0;
        for feller in &self.fellers {
            if feller.is_alive {
                count += 1;
            }
        }
        count
    }

    fn from_predecessors(predecessors: Population) -> Population {
        // compute a score for each feller
        // then sort them by descending score
        // and retain only the top 5%
        let mut scored_fellers = predecessors
            .fellers
            .into_iter()
            .map(|p| (score(&p), p))
            .collect::<Vec<_>>();
        scored_fellers.sort_by(|a, b| b.0.total_cmp(&a.0));
        let keep_len = (POPULATION_SIZE as f64 * 0.05).ceil() as usize;
        scored_fellers.truncate(keep_len);

        // compute the sum of all scores
        let mut score_sum = 0.0;
        for (score, _) in &scored_fellers {
            score_sum += score;
        }

        let mut descendants = vec![];
        let mut rng = StdRng::from_os_rng();

        // create 80% of the new descendants by random picking
        // while the highest scorers are most likely to procreate
        let procreation_len = (0.8 * POPULATION_SIZE as f64).ceil() as usize;
        while descendants.len() < procreation_len {
            let mut r = rng.random_range(0.0..score_sum);

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
