# もしもボックス九工大出張所

九州工業大学・大北研究室のオープンキャンパス向けデモです。インタビューから
未来の本人によるメッセージ動画を生成し、推論・入力データ・中間生成物を
研究室内JupyterHubで完結させます。

ターン単位の日本語ASR、インタビューLLM、台本設計VLM、本人声TTS、未来画像生成、
音声駆動動画生成を、交換可能な独立ワーカーとして接続しています。

## 別アカウントでの初回セットアップ

各メンバーは自分のJupyterHubへログインし、そのユーザーサーバーのターミナルで
実行します。

~~~bash
git clone <GitHub repository URL>
cd moshimo-box-kyutech

./scripts/bootstrap.sh
~~~

`bootstrap.sh`は次を行います。

- Conda環境を未作成の場合だけ作成
- Pythonとnpm依存関係の導入
- フロントエンドのビルド
- Git管理外の本番設定とローカルモデルカタログの作成
- 実行データ用ディレクトリの作成

次に、Hugging Face上で利用するgated modelの規約へ同意してから認証します。
トークンは各JupyterHubアカウント内へ保存され、Gitには入りません。

~~~bash
./scripts/huggingface-login.sh
~~~

推奨モデル一式を導入して診断します。モデル取得と環境構築は再実行可能で、
既存の環境とダウンロード済みファイルを再利用します。

~~~bash
./scripts/install-models.sh balanced
./scripts/doctor.sh
~~~

モデル構成は用途に合わせて選べます。

| 引数 | 導入内容 |
| --- | --- |
| `core` | ASRとインタビューLLMのみ |
| `fast` | `core`、Fish TTS、FLUX 4B、MuseTalk |
| `balanced` | `fast`、Qwen3-VL 8B、FLUX 9B。通常デモ向け |
| `full` | `balanced`、EchoMimicV3。全候補を利用可能にする |

詳しい引き継ぎ手順は
[研究室メンバー向けセットアップ](docs/lab-member-setup.md)を参照してください。

## 起動

~~~bash
./start-app.sh
./start-app.sh 8788
~~~

既定ポートは`8789`です。アプリは`127.0.0.1`だけにバインドされ、
現在ログインしているユーザーのJupyterHub認証済みプロキシURLを表示します。
各メンバーは表示されたURLをMicrosoft Edgeで開き、初回だけカメラとマイクを
許可します。

`config/default.yaml`の既定値は本番モードです。明示的に切り替える場合は、
アプリを停止して次を実行します。

~~~bash
./scripts/bootstrap.sh --production
./scripts/bootstrap.sh --debug
~~~

## 保存場所

既定では、リポジトリが`<workspace>/repositories/moshimo-box-kyutech`にある場合、
Conda環境を`<workspace>/env/moshimo-box-kyutech`へ作ります。

~~~text
<workspace>/env/moshimo-box-kyutech/
  app/
  generation/
  flux2-klein/
  musetalk/
  echomimic-v3/
~~~

別の場所を使う場合は、すべてのスクリプトで共通の環境変数を指定できます。

~~~bash
export MOSHIMO_ENV_ROOT=/path/to/moshimo-box-envs
export MOSHIMO_MODEL_ROOT=/path/to/shared-models
./scripts/bootstrap.sh
~~~

モデル重み、キャッシュ、ダウンロードした外部リポジトリ、Conda環境、
ローカル設定、ログ、入力映像・音声、生成物はGit管理外です。

## 設定

開発者設定の優先順位は次の通りです。

~~~text
config/default.yaml
  -> config/local.yaml
  -> .env / プロセス環境変数
  -> data/runtime/staff-settings.json
~~~

運営画面の変更は次のセッションから反映されます。モデル定義は次の二層です。

~~~text
config/model-catalog.yaml        共通のモデル情報・ライセンス・プリセット
config/model-catalog.local.yaml  導入・検証状態（Git管理外、自動生成）
~~~

環境変数による開発者設定は`MOSHIMO__SECTION__FIELD`形式です。

~~~bash
MOSHIMO__APP__DEBUG_MODE=false
MOSHIMO__SERVER__DEFAULT_PORT=8788
MOSHIMO__CAPTURE__FINALIZE_TIMEOUT_SECONDS=15
~~~

## 主な配置

~~~text
backend/app/contracts/          LLM/VLMの構造化出力契約
backend/app/workers/            共通ワーカーAPI、プロセス管理、アダプター
config/episodes/                1エピソード1 YAML
models/                         モデル重みと外部ソース（Git管理外）
data/sessions/<session_id>/     入力・中間生成物・完成物（終了時削除）
data/metrics/                   匿名の処理時間・メモリ統計（Git管理外）
scripts/                        構築、モデル導入、診断、起動補助
workers/envs/                   Conda環境の配置規約
~~~

設計は[開発設計メモ](docs/development-foundation.md)、モデル構成は
[生成モデル構成](docs/generation-models.md)を参照してください。

## 検証

通常の事前確認はモデルをロードしない軽量診断です。

~~~bash
./scripts/doctor.sh
~~~

開発時のテストは次で実行します。

~~~bash
./scripts/test-all.sh
./scripts/test-all.sh --e2e
~~~

E2Eを初めて実行するサーバーイメージでは、PlaywrightのChromiumとLinux依存を
別途準備する場合があります。通常のデモ運用ではPlaywrightは不要です。
