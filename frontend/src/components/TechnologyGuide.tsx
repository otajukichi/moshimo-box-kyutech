import {
  AudioWaveform,
  BookOpenText,
  BrainCircuit,
  Camera,
  FileText,
  Film,
  Image as ImageIcon,
  LockKeyhole,
  MessageCircleMore,
  MoveRight,
  Volume2,
  X
} from "lucide-react";
import { useEffect, useRef } from "react";
import "../technology-guide.css";

interface Props {
  open: boolean;
  onClose: () => void;
}

const modalities = [
  {
    key: "sound",
    label: "音声",
    detail: "波形・声色・話す速さ・間",
    icon: AudioWaveform
  },
  {
    key: "text",
    label: "文字",
    detail: "単語・文脈・指示・台本",
    icon: FileText
  },
  {
    key: "image",
    label: "画像",
    detail: "形・色・人物・空間配置",
    icon: Camera
  },
  {
    key: "video",
    label: "動画",
    detail: "連続する画像・動き・時間",
    icon: Film
  }
] as const;

const modelPatterns = [
  {
    index: "A",
    label: "一種類を扱う",
    name: "単一モダリティ",
    formula: "文字 → 文字",
    description: "同じ種類のデータを読み、同じ種類で答えます。文章だけで会話する言語モデルが代表例です。",
    example: "このデモ: 会話・台本を考えるAI",
    tone: "blue"
  },
  {
    index: "B",
    label: "種類を変換する",
    name: "モダリティ変換",
    formula: "音声 → 文字",
    description: "ある種類のデータを、別の種類へ写し替えます。音の並びと文字の並びの対応を学習しています。",
    example: "このデモ: 音声認識",
    tone: "cyan"
  },
  {
    index: "C",
    label: "複数を条件にする",
    name: "複数モダリティ入力",
    formula: "画像 + 文字 → 画像",
    description: "元画像の人物や構図を保ちながら、文字で指定された服装や背景へ変えます。",
    example: "このデモ: 未来画像の生成",
    tone: "yellow"
  },
  {
    index: "D",
    label: "複数を結び付けて生成する",
    name: "マルチモーダル生成",
    formula: "画像 + 音声 + 文字 → 動画",
    description: "見た目、声の時間、動きの指示を一つのモデル内で対応付け、時間を持つ別のデータを作ります。",
    example: "このデモ: 最終メッセージ動画",
    tone: "green"
  }
] as const;

const fusionSignals = [
  {
    label: "画像が伝えること",
    value: "誰が、どこにいるか",
    detail: "人物の顔、服装、背景、最初の姿勢を動画の基準にします。",
    icon: Camera,
    tone: "image"
  },
  {
    label: "音声が伝えること",
    value: "いつ、どのように話すか",
    detail: "発音の長さ、間、強弱に合わせて口や表情を時間方向に動かします。",
    icon: AudioWaveform,
    tone: "sound"
  },
  {
    label: "文字が伝えること",
    value: "どんな動きにするか",
    detail: "目線、身振り、カメラ、雰囲気など、音声だけでは分からない演出を指定します。",
    icon: FileText,
    tone: "text"
  }
] as const;

const pipeline = [
  {
    number: "01",
    name: "聞き取るAI",
    description: "音の波形を細かな特徴へ変換し、発音の並びに対応する日本語の文字を推定します。",
    input: "音声",
    output: "文字",
    badge: "CONVERSION",
    tone: "cyan",
    icon: AudioWaveform
  },
  {
    number: "02",
    name: "会話するAI",
    description: "文字になった会話と会話テーマを読み、直前の発言に関係する自然な返答を作ります。",
    input: "文字",
    output: "返答",
    badge: "TEXT",
    tone: "blue",
    icon: MessageCircleMore
  },
  {
    number: "03",
    name: "未来を設計するAI",
    description: "会話、抽選されたエピソード、映像モデルの条件を文章として読み、世界設定と台本を組み立てます。",
    input: "会話 + 設定",
    output: "設計・台本",
    badge: "TEXT",
    tone: "violet",
    icon: BrainCircuit
  },
  {
    number: "04",
    name: "未来を描くAI",
    description: "本人画像を見た目の基準にし、文字で指定した未来の服装、場所、表情を反映します。",
    input: "画像 + 文字",
    output: "未来画像",
    badge: "MULTI-INPUT",
    tone: "yellow",
    icon: ImageIcon
  },
  {
    number: "05",
    name: "声をつくるAI",
    description: "参照音声から声質を捉え、文字の台本に合わせて新しい発話音声を合成します。",
    input: "音声 + 文字",
    output: "未来の声",
    badge: "MULTI-INPUT",
    tone: "coral",
    icon: Volume2
  },
  {
    number: "06",
    name: "動かすAI",
    description: "未来画像、メッセージ音声、演出指示を同時に使い、互いの関係を保った動画を生成します。",
    input: "画像 + 音声 + 文字",
    output: "動画",
    badge: "MULTIMODAL",
    tone: "green",
    icon: Film
  }
] as const;

const glossary = [
  {
    term: "特徴表現",
    english: "FEATURE",
    description: "音の高さや画像の形などを、AIが計算できる数値の並びへ置き換えたもの。"
  },
  {
    term: "条件付け",
    english: "CONDITIONING",
    description: "画像や文字を手がかりとして与え、生成結果の人物、内容、動きを制御すること。"
  },
  {
    term: "対応付け",
    english: "ALIGNMENT",
    description: "音声のこの瞬間と口のこの形のように、異なるデータ同士の位置や時間を合わせること。"
  },
  {
    term: "生成",
    english: "GENERATION",
    description: "学習した対応関係を使い、入力条件に合う新しい文章、画像、音声、動画を作ること。"
  }
] as const;

export function TechnologyGuide({ open, onClose }: Props) {
  const closeButton = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (!open) return;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    closeButton.current?.focus();
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", closeOnEscape);
    return () => {
      document.body.style.overflow = previousOverflow;
      window.removeEventListener("keydown", closeOnEscape);
    };
  }, [onClose, open]);

  return (
    <section
      className={`technology-guide ${open ? "is-open" : ""}`}
      role="dialog"
      aria-modal="true"
      aria-labelledby="technology-guide-title"
      aria-hidden={!open}
      ref={(element) => element?.toggleAttribute("inert", !open)}
    >
      <header className="technology-guide-header">
        <div className="technology-guide-brand">
          <span>
            <BookOpenText size={19} />
          </span>
          <strong>TECHNOLOGY GUIDE</strong>
        </div>
        <span className="technology-guide-lab">KYUTECH / OKITA LAB.</span>
        <button
          ref={closeButton}
          className="icon-button"
          onClick={onClose}
          aria-label="技術解説を閉じる"
          title="閉じる"
        >
          <X size={22} />
        </button>
      </header>

      <div className="technology-guide-scroll">
        <section className="technology-guide-hero">
          <div className="technology-guide-inner">
            <span className="eyebrow">LESSON 01 / MODALITY</span>
            <h2 id="technology-guide-title">
              AIは、ことばだけを
              <br />
              見ているわけでは<span className="technology-mobile-line">ありません</span>
            </h2>
            <p>
              AIが扱う情報の種類を「モダリティ」と呼びます。文字、音声、画像、動画は、それぞれ構造の違うモダリティです。
              ファイルの拡張子の違いではなく、情報がどのように並び、どんな意味を持つかの違いを表しています。
            </p>

            <div className="modality-strip" aria-label="AIが扱う代表的なモダリティ">
              {modalities.map(({ key, label, detail, icon: Icon }) => (
                <div className={`modality-item modality-${key}`} key={key}>
                  <span className="modality-icon">
                    <Icon size={24} />
                  </span>
                  <span>
                    <strong>{label}</strong>
                    <small>{detail}</small>
                  </span>
                </div>
              ))}
            </div>
          </div>
        </section>

        <section className="technology-patterns">
          <div className="technology-guide-inner">
            <div className="guide-section-heading">
              <div>
                <span className="eyebrow">LESSON 02 / MODEL TYPES</span>
                <h3>入力と出力を見ると、モデルの役割が分かる</h3>
              </div>
              <p>
                「マルチモーダル」は単にAIを何個も使うという意味ではありません。
                一つのモデルが、どのモダリティを受け取り、何を出すかに注目します。
              </p>
            </div>

            <div className="model-pattern-list" aria-label="モダリティによるモデルの分類">
              {modelPatterns.map(({ index, label, name, formula, description, example, tone }) => (
                <article className={`model-pattern pattern-${tone}`} key={index}>
                  <span className="pattern-index">{index}</span>
                  <div className="pattern-name">
                    <small>{label}</small>
                    <strong>{name}</strong>
                  </div>
                  <div className="pattern-formula">{formula}</div>
                  <p>{description}</p>
                  <span className="pattern-example">{example}</span>
                </article>
              ))}
            </div>
            <p className="model-pattern-note">
              <strong>用語の幅:</strong>
              広い意味では、音声から文字への変換も複数モダリティをまたぐ技術です。
              このガイドでは違いを見やすくするため、「種類の変換」と「複数入力の統合」を分けて示しています。
            </p>
          </div>
        </section>

        <section className="technology-explainer">
          <div className="technology-guide-inner explainer-layout">
            <div className="explainer-copy">
              <span className="eyebrow">LESSON 03 / MULTIMODAL MODEL</span>
              <h3>一つのモデルの中で、異なる情報を関係づける</h3>
              <p>
                一つのAIモデルが複数のモダリティを入出力として扱う技術が、マルチモーダルAIです。
                モデルは音声、文字、画像をそのまま混ぜるのではなく、それぞれから特徴を取り出し、互いに対応する部分を結びつけます。
              </p>
              <p>
                このデモの最終動画モデルには、未来の本人画像、生成したメッセージ音声、動きの文字指示が入ります。
                三つを対応づけることで、「この人物が、このタイミングで、このように動く」という動画を作ります。
              </p>
              <div className="multimodal-definition">
                <strong>区別しよう</strong>
                <span>
                  このデモ全体は複数のAIモデルを接続したシステムです。それだけで、一つの巨大なマルチモーダルモデルになるわけではありません。
                  下の図は、その中にある一個の動画生成モデルを表しています。
                </span>
              </div>
            </div>
            <div className="multimodal-diagram" aria-label="一つのマルチモーダル動画モデルの概念図">
              <div className="diagram-inputs">
                <span className="diagram-sound"><AudioWaveform size={20} /> 音声の時間</span>
                <span className="diagram-text"><FileText size={20} /> 動きの指示</span>
                <span className="diagram-image"><Camera size={20} /> 人物の見た目</span>
              </div>
              <MoveRight size={28} aria-hidden="true" />
              <div className="diagram-core">
                <BrainCircuit size={38} />
                <strong>特徴を対応付ける</strong>
                <small>誰が・いつ・どう動く</small>
              </div>
              <MoveRight size={28} aria-hidden="true" />
              <div className="diagram-output">
                <Film size={30} />
                <strong>動画を生成</strong>
                <small>連続フレーム + 時間</small>
              </div>
            </div>
          </div>
        </section>

        <section className="technology-fusion">
          <div className="technology-guide-inner">
            <div className="guide-section-heading fusion-heading">
              <div>
                <span className="eyebrow">HOW IT CONNECTS</span>
                <h3>三つの入力は、別々の役割を持つ</h3>
              </div>
              <p>
                同じ動画を作る材料でも、画像、音声、文字が伝える情報は異なります。
                一つが欠けると、残りの入力からは決められない部分が増えます。
              </p>
            </div>

            <div className="fusion-layout">
              <div className="fusion-sources">
                {fusionSignals.map(({ label, value, detail, icon: Icon, tone }) => (
                  <article className={`fusion-source fusion-${tone}`} key={label}>
                    <span><Icon size={23} /></span>
                    <div>
                      <small>{label}</small>
                      <strong>{value}</strong>
                      <p>{detail}</p>
                    </div>
                  </article>
                ))}
              </div>
              <div className="fusion-result">
                <span className="fusion-result-icon"><Film size={34} /></span>
                <span className="eyebrow">ALIGNED OUTPUT</span>
                <h4>見た目と声を、時間の上で一致させる</h4>
                <p>
                  動画は静止画を何枚も時間順に並べたデータです。モデルは各時刻の顔や口の形を予測しながら、
                  元画像の人物らしさ、音声との同期、指示された動きが同時に保たれるよう生成します。
                </p>
                <dl>
                  <div><dt>空間</dt><dd>顔・服装・背景</dd></div>
                  <div><dt>時間</dt><dd>発音・瞬き・身振り</dd></div>
                  <div><dt>意味</dt><dd>台本・演出・雰囲気</dd></div>
                </dl>
              </div>
            </div>
          </div>
        </section>

        <section className="technology-pipeline">
          <div className="technology-guide-inner">
            <div className="pipeline-heading">
              <div>
                <span className="eyebrow">LESSON 04 / INSIDE THIS DEMO</span>
                <h3>未来の動画ができるまで</h3>
              </div>
              <p>
                一個の万能AIがすべてを行うのではなく、得意分野の違うモデルを順番に接続します。
                矢印の左が入力、右が出力です。
              </p>
            </div>
            <div className="technology-pipeline-grid">
              {pipeline.map(({ number, name, description, input, output, badge, tone, icon: Icon }) => (
                <article className={`pipeline-stage stage-${tone}`} key={number}>
                  <header>
                    <span>{number}</span>
                    <strong className="multimodal-stage-label">{badge}</strong>
                    <Icon size={25} />
                  </header>
                  <h4>{name}</h4>
                  <p>{description}</p>
                  <div className="pipeline-io">
                    <span>{input}</span>
                    <MoveRight size={15} />
                    <strong>{output}</strong>
                  </div>
                </article>
              ))}
            </div>
          </div>
        </section>

        <section className="technology-glossary">
          <div className="technology-guide-inner">
            <div className="guide-section-heading glossary-heading">
              <div>
                <span className="eyebrow">MINI GLOSSARY</span>
                <h3>図を見るための四つの言葉</h3>
              </div>
              <p>専門用語を、今回のデモで行っている処理に置き換えて整理します。</p>
            </div>
            <dl className="glossary-grid">
              {glossary.map(({ term, english, description }) => (
                <div key={term}>
                  <dt><span>{english}</span>{term}</dt>
                  <dd>{description}</dd>
                </div>
              ))}
            </dl>
          </div>
        </section>

        <section className="technology-local">
          <div className="technology-guide-inner local-layout">
            <span className="local-icon"><LockKeyhole size={29} /></span>
            <div>
              <span className="eyebrow">LOCAL AI</span>
              <h3>処理は研究室のサーバーの中で完結します</h3>
              <p>
                会話、音声、画像、生成途中のデータを外部の生成AIサービスへ送りません。
                工程が終わるたびに必要なモデルへ入れ替え、デモ終了後にはセッションのデータをまとめて削除します。
              </p>
            </div>
          </div>
        </section>
      </div>
    </section>
  );
}
