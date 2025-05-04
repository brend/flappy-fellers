use macroquad::{
    color::*,
    shapes::{draw_circle, draw_rectangle},
    window::{clear_background, next_frame, screen_height, screen_width},
};
use neural_network_study::NeuralNetwork;
use rand::prelude::*;

const HSPEED: f32 = 0.8;
const MAX_SPEED: f32 = 2.0;
const PIPE_PROBABILITY: f32 = 0.002;
const PIPE_WIDTH: f32 = 40.0;
const LIFT: f32 = 2.0;
const FELLER_X: f32 = 40.0;
const FELLER_R: f32 = 20.0;

#[macroquad::main("Flappy Feller")]
async fn main() {
    let mut rng = StdRng::from_os_rng();
    let nn = NeuralNetwork::new(5, 4, 2, Some(&mut rng));
    let mut pipes: Vec<Pipe> = vec![];
    let mut feller = Feller::new();

    loop {
        clear_background(WHITE);

        // spawn a new pipe with a certain probability
        if pipes.is_empty() || rng.random::<f32>() < PIPE_PROBABILITY {
            let spawn_allowed = match pipes.last() {
                Some(pipe) => pipe.x + 2.0 * PIPE_WIDTH < screen_width(),
                None => true,
            };
            if spawn_allowed {
                pipes.push(Pipe::random(&mut rng));
            }
        }

        // update pipes
        for pipe in pipes.iter_mut() {
            pipe.x -= HSPEED;
        }
        pipes.retain(|p| p.x + PIPE_WIDTH > 0.0);

        // update the feller based on the neural network's output
        // if is_key_pressed(KeyCode::Space) {
        //     feller.yspeed -= LIFT;
        // }
        if let Some(pipe) = pipes.first() {
            let w = screen_width();
            let h = screen_height();
            let input = vec![
                (feller.y / h) as f64,
                (feller.yspeed / MAX_SPEED) as f64,
                (pipe.x / w) as f64,
                (pipe.y1 / h) as f64,
                (pipe.y2 / h) as f64,
            ];
            let output = nn.predict(input);
            if output[0] > output[1] {
                feller.yspeed -= LIFT;
            }
        }

        // Update the feller's vertical speed with gravitation and clamping
        feller.yspeed = (feller.yspeed + 0.02).clamp(-MAX_SPEED, MAX_SPEED);
        feller.y = (feller.y + feller.yspeed).clamp(FELLER_R, screen_height() - FELLER_R);

        // draw pipes
        for pipe in &pipes {
            // check for collision
            let color = if (pipe.x - FELLER_X).abs() < FELLER_R
                && (feller.y - FELLER_R < pipe.y1 || feller.y + FELLER_R > pipe.y2)
            {
                RED
            } else {
                BLACK
            };

            draw_rectangle(pipe.x, 0.0, PIPE_WIDTH, pipe.y1, color);
            draw_rectangle(
                pipe.x,
                pipe.y2,
                PIPE_WIDTH,
                screen_height() - pipe.y2,
                color,
            );
        }

        // draw the feller
        draw_circle(FELLER_X, feller.y, FELLER_R, BLACK);

        next_frame().await
    }
}

struct Pipe {
    x: f32,
    y1: f32,
    y2: f32,
}

impl Pipe {
    pub fn random(rng: &mut StdRng) -> Pipe {
        let y1 = rng.random_range(50.0..200.0);
        let y2 = rng.random_range((y1 + 40.0)..(y1 + 180.0));
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
}

impl Feller {
    fn new() -> Feller {
        Feller {
            y: 0.0,
            yspeed: 0.0,
        }
    }
}
