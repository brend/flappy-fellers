use macroquad::{
    color::*,
    input::{KeyCode, is_key_pressed},
    shapes::{draw_circle, draw_rectangle},
    window::{clear_background, next_frame, screen_height, screen_width},
};
use rand::prelude::*;

const SPEED: f32 = 0.8;
const MAX_SPEED: f32 = 2.0;
const PIPE_PROBABILITY: f32 = 0.002;
const PIPE_WIDTH: f32 = 40.0;
const LIFT: f32 = 2.0;
const FELLER_X: f32 = 40.0;
const FELLER_R: f32 = 20.0;

#[macroquad::main("Flappy Feller")]
async fn main() {
    let mut rng = rand::rng();
    let mut pipes: Vec<Pipe> = vec![];
    let mut feller = Feller {
        y: 0.0,
        yspeed: 0.0,
    };

    loop {
        clear_background(WHITE);

        // spawn a new pipe with a certain probability
        if rng.random::<f32>() < PIPE_PROBABILITY {
            let y1 = rng.random_range(50.0..200.0);
            let y2 = rng.random_range((y1 + 40.0)..(y1 + 180.0));
            pipes.push(Pipe {
                x: screen_width(),
                y1,
                y2,
            })
        }

        // update pipes
        for pipe in pipes.iter_mut() {
            pipe.x -= SPEED;
        }
        pipes.retain(|p| p.x + PIPE_WIDTH > 0.0);

        // update the feller
        if is_key_pressed(KeyCode::Space) {
            feller.yspeed -= LIFT;
        }
        feller.yspeed = (feller.yspeed + 0.02).clamp(-MAX_SPEED, MAX_SPEED);
        feller.y += feller.yspeed;

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

struct Feller {
    y: f32,
    yspeed: f32,
}
