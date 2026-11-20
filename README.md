# 水下照片物理恢复工具（Underwater Photo Restoration）

为潜水中心学员水下照片交付自研的修图引擎。按**每张照片的深度与拍摄距离**，把水体
**衰减**与**后向散射分开计算**，并区分**闪光灯路径**与**环境光路径**，在去掉水体蓝绿色
的同时保留珊瑚本色、把学员肤色拉回正常人肤色范围。算法用 TypeScript 编写，计算密集部分
运行在 **WebGL2** 上。

## 为什么不是「拉曲线」

十米以下红光通道几乎只剩噪声。通用软件整体拉曲线会同时做三件错事：远处水体被拉紫、
近处人脸拉出假红、珊瑚原本的暖色被洗掉。本工具用 Jaffe–McGlamery 水下成像物理模型正向
描述退化，再按已知（或估计）的深度/距离做逆向求解，从根上区分不同光路径。

## 物理模型

每通道 k、像素 x，物距 d(x)、相机水深 z、单程透射 t = exp(−c·d)、下行环境光
E(z) = exp(−Kd·z)、闪光灯几何因子 G(x,d)：

```
M_k = ρ_k · [ E_k(z)·t_k  +  G·t_k² ]  +  V_k·(1 − t_k)·(1 + η·G)
        \___ 环境光（单程）___/  \_ 闪光灯（往返，故 t²）_/   \____ 后向散射光帷 ____/
```

- **衰减与后向散射分离**：ρ 的信号项与光帷 V·(1−t) 是两个独立项，逆向时分别扣除。
- **闪光灯/环境光分路径**：闪光灯与相机同位，光走往返故为 t²；环境光只走单程 t。
  闪光灯锥内近物与锥外远水在同一张照片里用各自的路径增益求解，不会近红远青。
- **Jerlov 水体类型**：10 种水型（I…III 开阔海域、1C…9C 近岸）各有 RGB 三通道衰减/
  后向散射系数与光帷颜色；红光衰减显著快于蓝光。
- **深度相关系数**：近表层颗粒负载使衰减偏大，按 12 m 尺度深度向 Jerlov 基线弛豫。

### 噪声处理（避免假红的关键）

深水红光通道信号低于传感器噪声时，逆向的 1/路径增益会把噪声放大成假红。引擎提供：

- `maxGain`：恢复增益硬上限；
- `regularization`：门控 Wiener 正则，仅在路径增益低于健康阈值（0.25）的低 SNR 通道
  （典型即深水红光）把反射率向灰先验收缩，健康通道与珊瑚像素保持精确反演；
- `autoRegularization()`：按红通道平均透射率自动选 λ。

## 肤色约束

- 用 BT.601 YCbCr 肤色椭圆（带软边）检测肤色，检测在编码 sRGB 空间进行；
- 仅对判定为肤色的像素，在 CIE Lab 内把色度投影到人类肤色色域（深肤色保持深、浅肤色
  保持浅，不压成同一种脸色）；
- 珊瑚/水体/植被在椭圆外，**逐位保持不动**（有测试守护）。

## 目录结构

```
src/
  color.ts        CIEDE2000（Sharma 公式）、sRGB 线性化、Lab/YCbCr
  water.ts        Jerlov 水型、深度相关衰减、下行辐照度
  strobe.ts       闪光灯锥几何与逆平方距离律
  model.ts        Jaffe–McGlamery 正向合成 / 标定逆向 / DCP 盲估计 / Wiener 正则
  skin.ts         肤色检测与 Lab 色域约束（CPU 参考实现）
  colorchart.ts   24 色 X-Rite ColorChecker 参考值
  scene.ts        程序化陆地场景 + 按物理模型合成水下测试图
  tone.ts         Reinhard 色调映射、sRGB 编解码
  shaders.ts      与 CPU 逐行对应的 GLSL ES 3.00 着色器
  webgl2.ts       WebGL2 分块管线（RGBA32F，控显存）、f32→f16 工具
  pipeline.ts     CPU/GPU 统一门面
tests/            vitest 单元/验收测试 + Playwright 真实 GPU 奇偶与性能测试
demo/             浏览器交互演示（合成图 vs 恢复图，可调水深/水型/闪光灯/盲估计）
scripts/          ΔE 基准、性能基准、demo 冒烟测试
```

## 使用

```bash
npm install
npm test                 # 全部 66 个测试（含真实 WebGL2 GPU 测试）
npm run bench            # ColorChecker ΔE 在多水深/水型下的数值证据
npm run perf             # 每像素成本与 30MP 外推
npm run demo             # 浏览器交互演示
npm run gl:install       # 仅首次：安装 Playwright Chromium（GPU 测试用）
```

### 代码入口

```ts
import { synthesizeUnderwater } from "./src/scene.js";
import { WebGL2Restorer } from "./src/webgl2.js";

const { underwater, params } = synthesizeUnderwater({
  width, height, depthM: 10, waterType: "IB", strobePower: 1,
});
const gl = offscreenCanvas.getContext("webgl2");
const restorer = new WebGL2Restorer(gl);
const { srgb, gpuMs } = restorer.restore(underwater, {
  params,                 // 标定模式带 distance；盲估计设 estimateDistance: true
  applySkin: true,
  encodeSrgb: true,
  tileSize: 1024,
});
```

## 验收指标对应关系

| 需求 | 实现与证据 |
| --- | --- |
| 按深度/距离分离衰减与后向散射 | `model.ts` 信号项与光帷项独立；`tests/acceptance.test.ts` |
| 闪光灯/环境光分路径，不近红远青 | 往返 t² vs 单程 t；`strobe/ambient split` 验收测试 |
| 去蓝绿但不洗掉珊瑚 | 肤色约束只作用于肤色像素；`coral colors are not washed out` 测试 |
| 肤色落在正常人肤色范围 | Lab 肤色域投影；`skin tones land in the human gamut` 测试 |
| TypeScript + WebGL2 | 全量 TS 严格模式；`webgl2.ts` + GLSL |
| 色卡恢复 ΔE < 5 | 标定往返 ΔE≈0；含传感器噪声+距离误差的鲁棒测试覆盖 I@5m–III@15m |
| 30MP ≤ 3 秒 | 分块 RGBA32F GPU 管线；GPU/CPU 奇偶误差 ~1e-7；`perf.ts` 预算核算 |
| 测试图按真实衰减模型合成陆地照片 | `scene.ts` 与逆向共用同一正向物理模型 |

### 关于 ΔE 的诚实边界

- **标定模式**（水深/水型/距离已知）：无噪声时数学上完全可逆，ΔE≈0；在真实传感器噪声
  （σ≈0.004）与 ±5% 距离误差下，I@5m 到 III@15m 的 24 色卡平均 ΔE 均 < 5（见
  `tests/robustness.test.ts` 与 `npm run bench`）。
- 极深/高浑浊水体（如 9C@20m）红光物理上已低于噪声底，任何算法都无法恢复出可靠红色——
  此时正则会输出自然的中性灰而非假红，这是物理约束下的正确行为，而非缺陷。
- **盲估计模式**给出正确的远近相对结构并锚定到物理工作距离，适合无测距元数据的场景；
  追求 ΔE<5 的精修交付应使用带距离图的标定模式。

## 工程备注

- 工作纹理采用 RGBA32F 而非 RGBA16F：水下红光线性值常 <0.05，半浮点量化经 1/路径增益
  放大后会产生可见误差（实测最大 ΔE 0.29）；32F 把 GPU/CPU 奇偶误差降到 ~1e-7。
- 大图按 1024px 分块处理，VRAM 占用不随总像素膨胀；每块纹理尺寸精确匹配子块，避免部分
  填充的坐标歧义。
