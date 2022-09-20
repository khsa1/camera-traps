use uuid::Uuid;
use zmq::Socket;
use event_engine::{plugins::Plugin, events::Event};
use event_engine::errors::EngineError;
use event_engine::events::EventType;
use crate::{events, config::errors::Errors};

use log::{info, error};
use std::{thread, time};

struct ImageGenPlugin {
    name: String,
    id: Uuid,
}

impl Plugin for ImageGenPlugin {

    /// The entry point for the plugin. The engine will start the plugin in its own
    /// thread and execute this function.  The pub_socket is used by the plugin to 
    /// publish new events.  The sub_socket is used by the plugin to get events 
    /// published by other plugins.
    fn start(
        &self,
        pub_socket: Socket,
        sub_socket: Socket,
    ) -> Result<(), EngineError> {

        // Announce our arrival.
        info!("{}", format!("{}", Errors::PluginStarted(self.name.clone(), self.get_id().to_hyphenated().to_string())));
        thread::sleep(time::Duration::new(1, 0));

        // Send our alive event.
        let ev = events::PluginStartedEvent::new(self.get_id().clone(), self.name.clone());
        let bytes = match ev.to_bytes() {
            Ok(v) => v,
            Err(e) => {
                // Log the error and abort if we can't serialize our start up message.
                let msg = format!("{}", Errors::EventToBytesError(self.name.clone(), ev.get_name(), e.to_string()));
                error!("{}", msg);
                return Err(EngineError::PluginExecutionError(self.name.clone(), self.get_id().to_hyphenated().to_string(), msg));
            } 
        };

        // Send the event serialization succeeded.
        match pub_socket.send(bytes, 0) {
            Ok(_) => (),
            Err(e) => {
                // Log the error and abort if we can't send our start up message.
                let msg = format!("{}", Errors::SocketSendError(self.name.clone(), ev.get_name(), e.to_string()));
                error!("{}", msg);
                return Err(EngineError::PluginExecutionError(self.name.clone(), self.get_id().to_hyphenated().to_string(), msg));
            }
        };

        // Enter our infinite work loop.
        loop {
            // Wait on the subsciption socket.
            let bytes = match sub_socket.recv_bytes(0) {
                Ok(b) => b,
                Err(e) => {
                    // We log error and then move on. It would probably be a good idea to 
                    // pause before continuing if there are too many errors in a short period
                    // of time.  This would avoid filling up the log file and burning cycles
                    // when things go sideways for a while.
                    let msg = format!("{}", Errors::SocketRecvError(self.name.clone(), e.to_string()));
                    error!("{}", msg);
                    continue;
                }
            };

            // Determine the event type.
            

            // Process the event.

            // Determine if we should break.
        }

    }

    /// Return the event subscriptions, as a vector of strings, that this plugin is interested in.
    fn get_subscriptions(&self) -> Result<Vec<Box<dyn EventType>>, EngineError> {
        Ok(vec![Box::new(events::PluginTerminateEvent::new(Uuid::new_v4(), String::from("*")))])
    }

    /// Returns the unique id for this plugin.
    fn get_id(&self) -> Uuid {self.id.clone()}
}

impl ImageGenPlugin {
    pub fn new() -> Self {
        ImageGenPlugin {
            name: "ImageGenPlugin".to_string(),
            id: Uuid::new_v4(),
        }
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn here_i_am() {
        println!("file test: image_gen_plugin.rs");
    }
}
