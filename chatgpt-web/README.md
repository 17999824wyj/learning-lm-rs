# ChatGPT 的仿写前端

多说无益，直接告诉使用方式。

## 前置要求

### Node

`node` 需要 `^16 || ^18 || ^19` 版本（`node >= 14` 需要安装 [fetch polyfill](https://github.com/developit/unfetch#usage-as-a-polyfill)），使用 [nvm](https://github.com/nvm-sh/nvm) 可管理本地多个 `node` 版本

```shell
node -v
```

### PNPM

如果你没有安装过 `pnpm`

```shell
npm install pnpm -g
```

## 安装依赖

在根目录下：

```shell
pnpm bootstrap
```

## 运行项目前端

在根目录下：

```shell
pnpm dev
```
