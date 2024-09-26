pub mod service {
    use crate::model::{self, Llama};
    use crate::schemas::{Error, Infer};
    use std::path::PathBuf;
    use std::sync::Arc;
    use tokenizers::Tokenizer;
    use tokio::sync::{
        mpsc::{self},
        Mutex,
    };

    pub struct Service {
        tokenizer: Box<Tokenizer>,
        model: Box<Llama<f32>>,
        pub default_sample: Mutex<SampleArgs>,
    }

    pub struct ServiceManager {
        service: Arc<Service>,
    }

    impl ServiceManager {
        pub fn infer(
            self: &Arc<Self>,
            Infer {
                inputs: messages,
                temperature,
                top_k,
                top_p,
            }: Infer,
        ) -> Result<mpsc::UnboundedReceiver<String>, Error> {
            let (sender, receiver) = mpsc::unbounded_channel();

            // 克隆 Arc<Self> 以便在异步任务中使用
            let self_clone = self.clone();

            // 异步执行推断任务
            tokio::spawn(async move {
                // 获取采样参数的副本
                let sample_args = {
                    let sample = self_clone.service.default_sample.lock().await;
                    SampleArgs {
                        temperature: temperature.unwrap_or(sample.temperature),
                        top_k: top_k.unwrap_or(sample.top_k),
                        top_p: top_p.unwrap_or(sample.top_p),
                        max_len: sample.max_len, // 暂定 max_len 不需要变动
                    }
                };

                // 调用异步 web_serve 方法并等待结果
                match self_clone.service.web_serve(&messages, sample_args).await {
                    Ok(output) => {
                        // 发送模型生成的输出到通道
                        if let Err(e) = sender.send(output) {
                            eprintln!("Failed to send output with error: {}", e);
                        }
                    }
                    Err(e) => {
                        // 发送错误到通道
                        if let Err(e) = sender.send(format!("Error: {:?}", e)) {
                            eprintln!("Failed to send error with error: {}", e);
                        }
                    }
                }
            });

            Ok(receiver)
        }

        pub fn new(service: Arc<Service>) -> Self {
            Self { service }
        }
    }

    #[derive(Clone, PartialEq, Debug)]
    pub struct SampleArgs {
        pub temperature: f32,
        pub top_k: usize,
        pub top_p: f32,
        pub max_len: usize,
    }

    impl Default for SampleArgs {
        #[inline]
        fn default() -> Self {
            Self {
                temperature: 0.,
                top_k: usize::MAX,
                top_p: 0.,
                max_len: 200,
            }
        }
    }

    impl Service {
        // 创建新的 Service 对象
        pub fn new(
            model_dir: &PathBuf,
            tokenizer_path: &PathBuf,
            default_sample: SampleArgs,
        ) -> Self {
            let llama = model::Llama::<f32>::from_safetensors(model_dir);
            let tokenizer = Tokenizer::from_file(tokenizer_path).unwrap();
            Self {
                tokenizer: Box::new(tokenizer),
                model: Box::new(llama),
                default_sample: default_sample.into(),
            }
        }

        // 创建默认配置的 Service 对象
        pub fn default() -> Self {
            let model_dir = &PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("models")
                .join("story");
            Self::new(
                model_dir,
                &model_dir.join("tokenizer.json"),
                SampleArgs::default(),
            )
        }

        pub async fn web_serve(
            self: &Arc<Self>,
            input: &str,
            sample: SampleArgs,
        ) -> Result<String, Error> {
            let binding = self.tokenizer.encode(input, true).unwrap();
            let input_ids = binding.get_ids();
            let output_ids = self.model.generate(
                input_ids,
                sample.max_len,
                sample.temperature,
                sample.top_k as u32,
                sample.top_p,
            );
            Ok(input.to_string() + &self.tokenizer.decode(&output_ids, true).unwrap())
        }
    }
}

use crate::response::{error, text_stream};
use crate::schemas::Error;
use http_body_util::{BodyExt, Empty};
use hyper::StatusCode;
use hyper::{
    body::{Bytes, Incoming},
    server::conn::http1,
    Method, Request, Response,
};
use hyper_util::rt::TokioIo;
use service::ServiceManager;
use std::{
    future::Future,
    net::{Ipv4Addr, SocketAddr, SocketAddrV4},
    pin::Pin,
    sync::Arc,
};
use tokio::net::TcpListener;
use tokio_stream::wrappers::UnboundedReceiverStream;

struct App(Arc<ServiceManager>);

impl Clone for App {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl hyper::service::Service<Request<Incoming>> for App {
    type Response = hyper::Response<http_body_util::combinators::BoxBody<Bytes, hyper::Error>>;
    type Error = hyper::Error;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn call(&self, req: Request<Incoming>) -> Self::Future {
        let manager = self.0.clone();

        macro_rules! response {
            ($method:ident; $f:expr) => {
                Box::pin(async move {
                    let whole_body = req.collect().await?.to_bytes();
                    println!("[TRACE] Request body >> {:?}", whole_body);
                    let req = serde_json::from_slice(&whole_body);
                    println!("[TRACE] Request body after decode >> {:?}", req);
                    Ok(match req {
                        Ok(req) => match manager.$method(req) {
                            Ok(ret) => {
                                println!("[TRACE] Request body will use >> {:?}", ret);
                                $f(ret)
                            }
                            Err(e) => error(e),
                        },
                        Err(e) => error(Error::WrongJson(e)),
                    })
                })
            };
        }

        match (req.method(), req.uri().path()) {
            (&Method::POST, "/infer") => {
                response!(infer; |ret| text_stream(UnboundedReceiverStream::new(ret)))
            }
            // Return 404 Not Found for other routes.
            _ => Box::pin(async move {
                println!(
                    "[INFO] {:?} {:?} has been stoped, because no resource found",
                    req.method(),
                    req.uri().path()
                );
                Ok(Response::builder()
                    .status(StatusCode::NOT_FOUND)
                    .body(
                        Empty::<Bytes>::new()
                            .map_err(|never| match never {})
                            .boxed(),
                    )
                    .unwrap())
            }),
        }
    }
}

pub async fn start_infer_service(service: service::Service, port: u16) -> std::io::Result<()> {
    let addr = SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::UNSPECIFIED, port));
    println!("[INFO] start service at {addr}");

    let app = App(Arc::new(ServiceManager::new(Arc::new(service))));
    let listener = TcpListener::bind(addr).await?;
    loop {
        let app = app.clone();
        let (stream, _) = listener.accept().await?;
        tokio::spawn(async move {
            if let Err(err) = http1::Builder::new()
                .serve_connection(TokioIo::new(stream), app)
                .await
            {
                println!("[ERROR] Error serving connection: {err:?}");
            }
        });
    }
}

#[test]
fn test_location() {
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let expected_project_name = "learning-lm-rs";

    // 分割目录路径并获取最后一个部分
    let last_part = project_dir
        .split_terminator("\\")
        .last()
        .unwrap_or_else(|| project_dir.split_terminator("/").last().unwrap());

    // 断言最后一个部分是否为预期值
    assert_eq!(last_part, expected_project_name);
}

#[tokio::test]
async fn test_service() {
    let service = std::sync::Arc::new(service::Service::default());
    let input = "Once upon a time";
    let output = service
        .web_serve(input, service::SampleArgs::default())
        .await
        .expect("Failed to serve web request");

    println!("{}", output);
}

#[tokio::test]
async fn test_infer() {
    // 创建 ServiceManager 实例
    let service_manager = Arc::new(ServiceManager::new(Arc::new(service::Service::default())));

    // 准备 Infer 请求
    let infer_request = crate::schemas::Infer {
        inputs: "Once upon a time".into(),
        temperature: Some(0.7),
        top_k: Some(40),
        top_p: Some(0.95),
    };

    // 调用 infer 方法
    let mut receiver = service_manager
        .infer(infer_request)
        .expect("Failed to call infer method");

    // 从接收器中读取输出
    let output = receiver.recv().await.expect("Failed to receive output");
    println!("[TRACE] Response body >> {:?}", output);
}

#[tokio::test]
async fn test_start_serve() {
    use reqwest::Client;
    use std::net::SocketAddr;
    use tokio::time::{sleep, timeout, Duration};

    // 设置一个端口号
    let port = 9001;
    let service = service::Service::default();
    let addr = SocketAddr::from(([127, 0, 0, 1], port));

    // 启动服务
    tokio::spawn(async move {
        if let Err(e) = start_infer_service(service, port).await {
            eprintln!("[ERROR] Failed to start service: {}", e);
        }
    });

    // 等待一段时间以确保服务启动
    sleep(Duration::from_secs(1)).await;

    // 创建 HTTP 客户端并尝试发送请求
    let client = Client::new();
    let url = format!("http://{}/infer", addr);

    // 准备要发送的数据
    let post_data = serde_json::json!({
        "inputs": "Once upon a time, ",
        "temperature": 0.7,
        "top_k": 40,
        "top_p": 0.95,
    });

    println!("[TRACE] Request body before deal >> {:?}", post_data);

    // 使用超时来限制请求等待时间, 10s 对应于 max_len 200
    let response = timeout(
        Duration::from_secs(10),
        client.post(&url).body(post_data.to_string()).send(),
    )
    .await;

    match response {
        Ok(Ok(resp)) => {
            // 检查响应状态码
            // assert_eq!(resp.status(), 200, "Expected status code 200");
            println!("[TRACE] Response status >> {:?}", resp.status());
            // 检查文本
            println!("[TRACE] Response body >> {:?}", resp.text().await);
        }
        Ok(Err(e)) => {
            eprintln!("[ERROR] Request failed: {}", e);
            panic!("Request failed");
        }
        Err(_) => {
            eprintln!("[ERROR] Request timed out");
            panic!("Request timed out");
        }
    }
}
