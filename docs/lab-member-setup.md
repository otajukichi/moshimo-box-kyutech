# 研究室メンバー向けセットアップ

この手順は、各メンバーが自分のJupyterHubユーザーサーバーでアプリを起動し、
手元のMicrosoft Edgeから実演するためのものです。モデルは手元のPCではなく
JupyterHub側のGPUで動きます。

## 事前に必要なもの

- 研究室JupyterHubへログインできるアカウント
- NVIDIA GPUが割り当てられたユーザーサーバー
- リポジトリのURL
- Hugging Faceアカウント
- 利用するgated modelの規約同意
- Microsoft Edgeで利用できるカメラとマイク
- モデルを保存できるサーバー側ストレージ

Hugging Faceトークンをチャット、GitHub、設定ファイルへ貼り付けないでください。

## 初回だけ行う操作

JupyterHubのターミナルで実行します。

~~~bash
git clone <GitHub repository URL>
cd moshimo-box-kyutech
./scripts/bootstrap.sh
~~~

ブラウザで利用モデルの公式Hugging Faceページを開き、必要な規約へ同意します。
その後、ターミナルへ戻って認証します。

~~~bash
./scripts/huggingface-login.sh
~~~

通常デモ用のモデルを導入します。途中で通信が切れても、同じコマンドを再実行
すれば環境とダウンロード済みファイルを再利用します。

~~~bash
./scripts/install-models.sh balanced
./scripts/doctor.sh
~~~

GPT-OSS 20Bをインタビュー・計画LLMの選択肢へ加える場合は、次を実行します。

~~~bash
./scripts/install-models.sh gpt-oss
./scripts/doctor.sh
~~~

EchoMimicV3とGPT-OSSを含む全候補をまとめて導入する場合は、次を実行します。

~~~bash
./scripts/install-models.sh full
./scripts/doctor.sh
~~~

## デモ当日の起動

~~~bash
cd <cloneした場所>/moshimo-box-kyutech
./scripts/doctor.sh
./start-app.sh
~~~

ターミナルへ表示されたURLをMicrosoft Edgeで開きます。他人のURLを使わず、
必ず自分のJupyterHubユーザー名を含むURLを使ってください。

初回アクセス時は、Edgeのサイト権限でカメラとマイクを許可します。映像が暗い、
別のマイクが選ばれる、音が入らない場合は、アドレスバー左側のサイト権限と
Windowsの入力デバイスを確認します。

アプリを終了するまで起動ターミナルを閉じないでください。終了はターミナルで
`Ctrl+C`です。

## 更新

コード更新後は次を実行します。

~~~bash
git pull
./scripts/bootstrap.sh
./scripts/activate-installed-models.sh
./scripts/doctor.sh
~~~

モデルIDやリビジョンが変更されたときだけ、改めて次を実行します。

~~~bash
./scripts/install-models.sh balanced
# GPT-OSSを利用している場合
./scripts/install-models.sh gpt-oss
~~~

`bootstrap.sh`を再実行してもConda環境は削除・再作成されません。

## よくある停止原因

### URLを開けない

起動ターミナルに表示されたURLをそのまま使います。ポートが使用中なら別の番号で
起動します。

~~~bash
./start-app.sh 8791
~~~

### カメラまたはマイクを利用できない

Edgeのサイト権限、Windowsのプライバシー設定、入力デバイスを確認します。
JupyterHubのプロキシURLはユーザーごとに異なるため、権限も各メンバーが一度ずつ
許可します。

### モデルが見つからない

~~~bash
./scripts/activate-installed-models.sh
./scripts/doctor.sh
~~~

不足が表示された場合は、対応するモデル構成を再実行します。

~~~bash
./scripts/install-models.sh balanced
~~~

### デバッグ表示が出る

アプリを停止して本番設定へ戻します。

~~~bash
./scripts/bootstrap.sh --production
./start-app.sh
~~~

### ワーカーが残ってGPUを使用している

まず起動中のアプリを`Ctrl+C`で終了します。復旧しない場合はJupyterHubの
ユーザーサーバーを停止・再起動してから、`doctor.sh`を実行します。

## Gitへ含まれないもの

次の内容はcloneでは取得できません。

- Conda環境
- npm依存関係とフロントエンドのビルド結果
- Hugging Face認証
- モデル重みとダウンロード済み外部リポジトリ
- `config/local.yaml`
- `config/model-catalog.local.yaml`
- 運営画面で保存した設定
- セッションの映像、音声、文字起こし、画像、音声、動画
- ログと匿名メトリクス
- Edgeのカメラ・マイク許可

このうちEdgeの権限とモデル規約への同意以外は、リポジトリ内のスクリプトで
再作成または診断できます。

## GitHub公開前

- プロジェクトのライセンスを決定する
- モデル重みや外部リポジトリをコミットしない
- Hugging Faceトークンや認証キャッシュをコミットしない
- Git履歴にも秘密情報や個人データがないことを確認する
- `NOTICE`と各モデルの利用条件を確認する
- `./scripts/doctor.sh`が通る新規ユーザーで一度通し実演する
