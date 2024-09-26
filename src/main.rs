mod config;
mod kvcache;
mod model;
mod operators;
mod params;
mod response;
mod schemas;
mod service;
mod tensor;

use service::service::Service;
use service::start_infer_service;

#[tokio::main]
async fn main() {
    // 设置端口号
    let port = 9001;

    // 创建 Service 实例
    let service = Service::default();

    // 启动推断服务
    if let Err(e) = start_infer_service(service, port).await {
        eprintln!("Failed to start infer service: {}", e);
    }
}
