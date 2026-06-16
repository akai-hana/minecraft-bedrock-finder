/// bedrockformation
/// Rust port of the Minecraft Bedrock Formation Finder.
///
/// Usage: bedrockformation <seed> <x:z> <floor|roof> [x,y,z:bedrock ...]
/// Example: bedrockformation 124352345 0:0 floor 0,-63,0:1 1,-62,0:1 0,-63,1:0

mod core;
mod gui;

use iced::{Application, Settings, window};

fn main() -> iced::Result {
    gui::App::run(Settings {
        window: window::Settings {
            size: iced::Size::new(800.0, 650.0),
            min_size: Some(iced::Size::new(620.0, 400.0)),
            ..Default::default()
        },
        ..Default::default()
    })
}
