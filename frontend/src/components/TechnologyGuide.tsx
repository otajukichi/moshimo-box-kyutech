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
    detail: "ことば・声色・間",
    icon: AudioWaveform
  },
  {
    key: "text",
    label: "文字",
    detail: "会話・意味・物語",
    icon: FileText
  },
  {
    key: "image",
    label: "画像",
    detail: "顔・服装・背景",
    icon: Camera
  },
  {
    key: "video",
    label: "動画",
    detail: "動き・表情・時間",
    icon: Film
  }
] as const;

const pipeline = [
  {
    number: "01",
    name: "聞き取るAI",
    description: "マイクの音を解析し、話した内容を日本語の文字へ変換します。",
    input: "音声",
    output: "文字",
    tone: "cyan",
    icon: AudioWaveform
  },
  {
    number: "02",
    name: "会話するAI",
    description: "これまでの会話を覚え、次に聞くと面白そうなことを考えます。",
    input: "文字",
    output: "質問",
    tone: "blue",
    icon: MessageCircleMore
  },
  {
    number: "03",
    name: "未来を設計するAI",
    description: "会話とエピソードを組み合わせ、未来の世界と台本を組み立てます。",
    input: "会話 + 設定",
    output: "設計・台本",
    tone: "violet",
    icon: BrainCircuit
  },
  {
    number: "04",
    name: "未来を描くAI",
    description: "本人の画像と未来の設計を読み、未来の姿と背景を描きます。",
    input: "画像 + 文字",
    output: "未来画像",
    tone: "yellow",
    icon: ImageIcon
  },
  {
    number: "05",
    name: "声をつくるAI",
    description: "インタビューの声を参考に、未来の本人が台本を読む声を合成します。",
    input: "音声 + 台本",
    output: "未来の声",
    tone: "coral",
    icon: Volume2
  },
  {
    number: "06",
    name: "動かすAI",
    description: "未来画像、メッセージ音声、演出の文字指示を同時に読み、時間を持つ映像へ変換します。",
    input: "画像 + 音声 + 文字",
    output: "動画",
    tone: "green",
    icon: Film
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
            <span className="eyebrow">ONE MODEL, MULTIPLE MODALITIES</span>
            <h2 id="technology-guide-title">
              AIは、ことばだけを
              <br />
              見ているわけでは<span className="technology-mobile-line">ありません</span>
            </h2>
            <p>
              人が音を聞き、景色を見て、ことばを考えるように、AIにも異なる種類のデータを扱うものがあります。
              今回の中心技術は、一つのモデルが文字・音声・画像を一緒に受け取り、動画という別の種類のデータをつくることです。
            </p>

            <div className="modality-strip" aria-label="AIが扱うデータの種類">
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

        <section className="technology-explainer">
          <div className="technology-guide-inner explainer-layout">
            <div className="explainer-copy">
              <span className="eyebrow">MULTIMODAL AI</span>
              <h3>一つのモデルが、種類の違う情報を関係づける</h3>
              <p>
                音声、文字、画像、動画のようなデータの種類を「モダリティ」と呼びます。
                一つのAIモデルが複数のモダリティを入出力として扱う技術が、マルチモーダルAIです。
              </p>
              <p>
                このデモの動画生成モデルには、未来の姿を描いた画像、メッセージ音声、動きや場面を示す文字を入力します。
                三つを同時に関係づけることで、「その人が、その内容を、その声で話す動画」へ変換します。
              </p>
              <div className="multimodal-definition">
                <strong>ここがポイント</strong>
                <span>
                  音声AI、画像AI、動画AIが別々に並んでいるだけでは、一つのマルチモーダルモデルとは呼びません。
                  同じモデルの中で複数種類の情報を結びつけて処理することが重要です。
                </span>
              </div>
            </div>
            <div className="multimodal-diagram" aria-label="マルチモーダルAIの概念図">
              <div className="diagram-inputs">
                <span className="diagram-sound"><AudioWaveform size={20} /> 音声</span>
                <span className="diagram-text"><FileText size={20} /> 文字</span>
                <span className="diagram-image"><Camera size={20} /> 画像</span>
              </div>
              <MoveRight size={28} aria-hidden="true" />
              <div className="diagram-core">
                <BrainCircuit size={38} />
                <strong>同時に解釈</strong>
                <small>関係を保つ</small>
              </div>
              <MoveRight size={28} aria-hidden="true" />
              <div className="diagram-output">
                <Film size={30} />
                <strong>動画を生成</strong>
                <small>動き・表情・時間</small>
              </div>
            </div>
          </div>
        </section>

        <section className="technology-pipeline">
          <div className="technology-guide-inner">
            <div className="pipeline-heading">
              <div>
                <span className="eyebrow">INSIDE THIS DEMO</span>
                <h3>未来の動画ができるまで</h3>
              </div>
              <p>
                工程ごとのAIと、その途中で複数モダリティを扱うモデルの関係を示しています。
              </p>
            </div>
            <div className="technology-pipeline-grid">
              {pipeline.map(({ number, name, description, input, output, tone, icon: Icon }) => (
                <article className={`pipeline-stage stage-${tone}`} key={number}>
                  <header>
                    <span>{number}</span>
                    {Number(number) >= 4 && (
                      <strong className="multimodal-stage-label">MULTIMODAL</strong>
                    )}
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

        <section className="technology-local">
          <div className="technology-guide-inner local-layout">
            <span className="local-icon"><LockKeyhole size={29} /></span>
            <div>
              <span className="eyebrow">LOCAL AI</span>
              <h3>処理は研究室のサーバーの中で完結します</h3>
              <p>
                会話、音声、画像、生成途中のデータを外部の生成AIサービスへ送りません。
                工程が終わるたびにAIを入れ替え、デモ終了後にはセッションのデータをまとめて削除します。
              </p>
            </div>
          </div>
        </section>
      </div>
    </section>
  );
}
