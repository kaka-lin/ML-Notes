# MediaPipe Learning Notes

MediaPipe 是 Google Research 所開發的多媒體機器學習模型應用框架。透過 MediaPipe，可以輕鬆地將模型（如:物件偵測、手部追蹤、姿態檢測）部署到移動設備（Android、iOS）、Web、桌面、嵌入式平台或後端伺服器等。

MediaPipe 目前支援 `Java(Android)`、`Python`、`Javascript/TypeScript`、`C++` 等語言

## Get Start

### Setup guide for Python

Building applications with MediaPipe Tasks requires the following development environment resources:

- Python 3.7-3.10
- PIP 19.0 or higher (>20.3 for macOS)
- For Macs using Apple silicon M1 and M2 chips, use the Rosetta Translation Environment. See the [Apple documentation](https://developer.apple.com/documentation/apple-silicon/about-the-rosetta-translation-environment/) for more information on Rosetta setup and usage.

Install the `MediaPipe package`:

```bash
$ pip install install mediapipe
```

After installing the package, import it into your development project.

```python3
import mediapipe as mp
```

## Examples
