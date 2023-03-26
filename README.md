# SquarePhotoFrame

撮影した写真に画像にあった色の枠を足して正方形にします。

## 使い方

- 必要なライブラリのインストール
  - `pip install -r requirements.txt`
- フォルダの中身のjpgファイルを一括で変換する場合
  - `python frame_maker.py -d {対象のディレクトリ} {出力先のディレクトリ}`
- 特定のjpgファイルを一件変換する場合
  - `python frame_maker.py -f {対象のファイル} {出力先のディレクトリ}`