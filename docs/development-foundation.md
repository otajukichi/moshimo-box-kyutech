# 開発基盤設計

## 方針

- 1台のPC、運営スタッフ操作、同時セッション1件
- 外部生成AI APIは使わず、入力・中間生成物・ログを研究室サーバー内へ限定
- 実モデルはカタログとアダプターで交換し、UIや生成フローから固有APIを隔離
- GPU工程は同時実行せず、独立プロセス終了後に次モデルをロード
- 設定は共通設定、マシン固有設定、運営設定の三層へ分離

## アプリ状態

```text
ワーカー事前準備
  -> タイトル
  -> consent
  -> device_check
  -> conversation
  -> generating
  -> review
  -> 終了・全データ削除・事前準備

任意の処理
  -> error / emergency stop
  -> 処理停止・全データ削除・事前準備
```

FastAPIが状態の正本で、Reactは`GET /api/session/current`をポーリングします。
インタビュー中のページ離脱は即削除、生成中の再読込は同じセッションへ復帰します。
サーバー再起動時には未完了セッションを削除します。

設定はセッション開始時にスナップショットされます。進行中に運営設定を変更しても
現在のセッションへは影響せず、次のセッションから反映されます。

## 16ワーカーと4プロセス群

| プロセス群 | 交換可能な役割 |
| --- | --- |
| interview | `audio_preprocess_worker`, `streaming_asr_worker`, `interview_llm_worker`, `interview_tts_worker` |
| material_preparation | `final_asr_worker`, `interview_summary_worker`, `episode_selector`, `script_design_llm_worker`, `script_safety_review_worker` |
| generation | `reference_frame_selector`, `voice_reference_selector`, `image_generation_worker`, `voice_clone_tts_worker` |
| finishing | `video_generation_worker`, `lip_sync_worker`, `video_postprocess_worker` |

これは16個の常駐サービスを意味しません。16個は差し替え境界、4群は起動・解放の
管理単位です。タイトル表示前にinterview群を準備し、インタビュー終了後に解放、
生成工程は原則1役割ずつ起動・実行・終了します。

## 共通ワーカー契約

`backend/app/workers/base.py`の`WorkerAdapter`が次を定義します。

- `load`
- `healthcheck`
- `run`
- `cancel`
- `unload`

全リクエスト・レスポンスには`schema_version`、worker role、backend、model ID、
revision、dtype、quantization、device、入出力パス、期限、構造化エラー、処理時間、
peak CPU/GPU memoryを持たせます。プロセス間通信はlocalhost HTTPだけで、起動ごとに
生成する秘密キーをヘッダーへ付けます。ポートは外部公開しません。

モデル固有コードは`backend/app/workers/adapters/`へ置き、
`module.path:create_worker`形式のfactoryを`adapter_entrypoint`へ登録します。依存関係が
衝突するモデルは、そのモデル用Conda環境のPythonと共通runtimeコマンドを指定します。

```yaml
- id: "example-model"
  roles: ["image_generation_worker"]
  backend: "example"
  model_id: "organization/model"
  revision: "exact-revision"
  dtype: "bfloat16"
  quantization: "none"
  device: "cuda:0"
  environment: "image-example"
  python_bin: "image-example/bin/python"
  command: ["-m", "backend.app.workers.runtime"]
  adapter_entrypoint: "backend.app.workers.adapters.example:create_worker"
  model_path: "image/example-model"
  timeout_seconds: 300
  installed: false
  validated: false
  last_healthcheck: "not_run"
```

運営画面には`installed: true`、`validated: true`、`last_healthcheck: passed`で、パスと
Pythonが存在するモデルだけを表示します。

## LLMの構造化契約

`backend/app/contracts/`にPydanticモデルを置き、JSON Schemaを生成できます。

- `InterviewTurnInput` / `InterviewTurnOutput`
- `ScriptDesignInput` / `ScriptDesignOutput`
- `ScriptSafetyReviewOutput`

インタビュー出力は取得情報、質問済み項目、次の深掘り、本人発話文字数、経過時間、
終了判断・理由、次の発話を保持します。台本設計出力は未来世界、未来の本人、肯定的な
解釈、衣装、背景、カメラ、感情、ナレーション、ショット、画像・動画・音声指示、
安全メモ、フォールバックを検証可能な形で保持します。台本設計ではモデルに一括JSONを
要求せず、各項目を平文で段階生成した後にアプリ側でこの契約へ格納します。

## モデルカタログとConda環境

- `config/model-catalog.yaml`: 候補、revision、量子化、ライセンス、3プリセット
- `config/model-catalog.local.yaml`: このマシンのパスと導入・実測済み状態
- `models/`: 重み、キャッシュ、ダウンロードしたモデルリポジトリ
- `<workspace>/env/moshimo-box-kyutech/app`: 基本環境
- `<workspace>/env/moshimo-box-kyutech/<model-or-family>`: 競合時だけ追加

モデル候補調査は次段階です。公式リポジトリ、公式モデルカード、公式ドキュメントで
revision、VRAM、言語、コード・重みライセンス、研究デモ条件を確認してから共通カタログへ
追加し、このJupyterHubで実測後にローカルカタログを`passed`へ変更します。

## エピソード

`config/episodes/*.yaml`を1ファイル1エピソードとして読み込みます。システムは生成せず、
既存ファイルの有効化・フィルタ・抽選だけを行います。

- formal: `formal_mode_allowed`かつ`public_demo_allowed`、`limited_only: false`のみ
- underground: 有効な全エピソード
- formal: 台本生成後に`safety_review_worker`を実行
- underground: 追加LLMジャッジを省略

基本レアリティを先に抽選し、`config/effects.yaml`の演出を別抽選します。未来からのSOS等は
`rarity_upgrade_steps`で最終レアリティを上げます。

## 収録と一時データ

カメラ確認開始からインタビュー終了まで映像を約5秒単位で録画し、回答音声はターンごとに
別ファイルへ保存します。ブラウザは未送信キュー、再試行、200MB上限を持ちます。無音終了は
既定1.8秒、回答上限は30秒です。文字起こしはAIワーカー接続後に本人発話だけを集計します。

```text
data/sessions/<session_id>/
  session.json
  input/
    transcript.json
    interview-state.json
    audio/answers/
    video/chunks/
  intermediate/<worker_role>/
  output/
```

セッション本文、音声、映像、プロンプト、生成物はログへ出しません。`data/metrics`のSQLiteは
ランダムなセッションID、モデル情報、成功可否、時間、メモリだけを保持します。

## 時間制限とフォールバック

生成時間はインタビュー終了後から計測し、既定30分です。設定値超過時は現在タスクをキャンセル、
ワーカープロセスを終了、一時データを削除し、時間超過画面を表示します。

- 同一モデルの再起動: 開発者設定回数まで
- 軽量モデルへの自動切替: 運営設定、既定ON、カタログにfallbackがある場合のみ1段
- 簡易動画フォールバック: 運営設定、既定OFF

現段階のスタブは最終動画を偽造せず、`video-placeholder.json`を出力します。

## 現在の未接続範囲

- ASRと自然なインタビューLLM
- インタビュー用TTS
- 要約、未来設定、台本、安全レビューの実モデル
- 参照フレーム・参照音声の品質選定
- 本人画像、本人声TTS、動画、リップシンク、後処理
- AI生成ラベルの動画への焼き込み

カメラ・マイク収録経路、UI、設定、セッション削除、ワーカー起動・認証・停止、スタブ進捗、
構造化契約は実装済みです。
