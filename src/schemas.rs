use hyper::StatusCode;
use std::fmt::Debug;

#[derive(serde::Deserialize)]
pub(crate) struct Infer {
    pub inputs: String,
    pub temperature: Option<f32>,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
}

impl Debug for Infer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Infer")
            .field("inputs", &self.inputs)
            .field("temperature", &self.temperature)
            .field("top_k", &self.top_k)
            .field("top_p", &self.top_p)
            .finish()
    }
}

pub(crate) trait Success {
    fn msg(&self) -> &str;
}

#[derive(Debug)]
pub(crate) enum Error {
    SessionBusy,
    SessionNotFound,
    WrongJson(serde_json::Error),
}

#[derive(serde::Serialize)]
struct ErrorBody {
    status: u16,
    code: u16,
    message: String,
}

impl Error {
    #[inline]
    pub const fn status(&self) -> StatusCode {
        match self {
            Self::SessionNotFound => StatusCode::NOT_FOUND,
            Self::SessionBusy => StatusCode::NOT_ACCEPTABLE,
            Self::WrongJson(_) => StatusCode::BAD_REQUEST,
        }
    }

    #[inline]
    pub fn body(&self) -> serde_json::Value {
        macro_rules! error {
            ($code:expr, $msg:expr) => {
                ErrorBody {
                    status: self.status().as_u16(),
                    code: $code,
                    message: $msg.into(),
                }
            };
        }

        #[inline]
        fn json(v: impl serde::Serialize) -> serde_json::Value {
            serde_json::to_value(v).unwrap()
        }

        match self {
            Self::SessionNotFound => json(error!(0, "Session not found")),
            Self::SessionBusy => json(error!(0, "Session is busy")),
            Self::WrongJson(e) => json(error!(0, e.to_string())),
        }
    }
}
