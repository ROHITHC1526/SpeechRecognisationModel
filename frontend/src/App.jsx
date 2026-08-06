import { useEffect, useMemo, useRef, useState } from "react";
import {
  Button,
  Card,
  Empty,
  Input,
  List,
  Progress,
  Select,
  Space,
  Typography,
  Upload,
  notification,
} from "antd";
import {
  AudioOutlined,
  DeleteOutlined,
  DownloadOutlined,
  HistoryOutlined,
  MoonOutlined,
  PauseCircleOutlined,
  PlayCircleOutlined,
  ReloadOutlined,
  SoundOutlined,
  SunOutlined,
  UploadOutlined,
  VideoCameraOutlined,
} from "@ant-design/icons";
import axios from "axios";
import { jsPDF } from "jspdf";
import { useTranslation } from "react-i18next";
import "antd/dist/reset.css";
import "./App.css";
import "./i18n";

const { Title, Text, Paragraph } = Typography;

const emotionProfiles = {
  happy: {
    emoji: "😊",
    color: "#ff8e53",
    icon: "✨",
    description: "You sound cheerful and positive.",
  },
  sad: {
    emoji: "😢",
    color: "#4f7cff",
    icon: "🌧️",
    description: "The speech expresses sadness.",
  },
  angry: {
    emoji: "😠",
    color: "#ff4d4f",
    icon: "🔥",
    description: "The speech contains anger.",
  },
  calm: {
    emoji: "😌",
    color: "#2ec4b6",
    icon: "🌿",
    description: "The speech sounds relaxed.",
  },
  fearful: {
    emoji: "😨",
    color: "#8b5cf6",
    icon: "🌙",
    description: "The speech reflects fear or caution.",
  },
  disgust: {
    emoji: "🤢",
    color: "#6b7280",
    icon: "🧪",
    description: "The speech conveys disgust.",
  },
  surprised: {
    emoji: "😲",
    color: "#f59e0b",
    icon: "⚡",
    description: "The speech feels unexpected or amazed.",
  },
  neutral: {
    emoji: "😐",
    color: "#64748b",
    icon: "🫥",
    description: "The speech appears balanced and neutral.",
  },
};

const formatFileSize = (size) => {
  if (!size) return "0 KB";
  const units = ["B", "KB", "MB", "GB"];
  let i = 0;
  let value = size;
  while (value >= 1024 && i < units.length - 1) {
    value /= 1024;
    i += 1;
  }
  return `${value.toFixed(value < 10 && i > 0 ? 1 : 0)} ${units[i]}`;
};

const normalizeEmotion = (value) => {
  if (!value) return "neutral";
  const normalized = String(value).toLowerCase();
  if (normalized.includes("happy")) return "happy";
  if (normalized.includes("sad")) return "sad";
  if (normalized.includes("angry")) return "angry";
  if (normalized.includes("calm")) return "calm";
  if (normalized.includes("fear")) return "fearful";
  if (normalized.includes("disgust")) return "disgust";
  if (normalized.includes("surpr")) return "surprised";
  return "neutral";
};

const buildProbabilityData = (predictedEmotion, confidence, rawProbabilities) => {
  const fallback = Object.entries(emotionProfiles).map(([key]) => ({
    emotion: key,
    confidence: key === normalizeEmotion(predictedEmotion) ? confidence : 0,
  }));

  if (!rawProbabilities) return fallback;

  if (Array.isArray(rawProbabilities)) {
    return rawProbabilities.map((entry) => {
      const emotionName = entry.emotion || entry.label || entry.name || "neutral";
      const probability = entry.confidence ?? entry.value ?? entry.probability ?? 0;
      return {
        emotion: String(emotionName),
        confidence: Number(probability),
      };
    });
  }

  if (typeof rawProbabilities === "object") {
    return Object.entries(rawProbabilities).map(([emotion, probability]) => ({
      emotion,
      confidence: Number(probability),
    }));
  }

  return fallback;
};

function App() {
  const { t, i18n } = useTranslation();
  const audioRef = useRef(null);
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [emotion, setEmotion] = useState("");
  const [confidence, setConfidence] = useState(0);
  const [loading, setLoading] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [volume, setVolume] = useState(1);
  const [isPlaying, setIsPlaying] = useState(false);
  const [theme, setTheme] = useState("light");
  const [history, setHistory] = useState([]);
  const [search, setSearch] = useState("");
  const [probabilities, setProbabilities] = useState([]);
  const [dragActive, setDragActive] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [recordedBlob, setRecordedBlob] = useState(null);
  const [recordedUrl, setRecordedUrl] = useState("");
  const [mediaRecorder, setMediaRecorder] = useState(null);
  const [recordedChunks, setRecordedChunks] = useState([]);

  const selectedProfile = useMemo(() => {
    const profile = emotionProfiles[normalizeEmotion(emotion)] || emotionProfiles.neutral;
    return profile;
  }, [emotion]);

  const confidenceLevel = confidence >= 85 ? "High" : confidence >= 60 ? "Medium" : "Low";

  useEffect(() => {
    const savedTheme = localStorage.getItem("theme") || "light";
    const savedLanguage = localStorage.getItem("language") || "en";
    const savedHistory = localStorage.getItem("history");
    setTheme(savedTheme);
    setHistory(savedHistory ? JSON.parse(savedHistory) : []);
    i18n.changeLanguage(savedLanguage);
  }, [i18n]);

  useEffect(() => {
    document.body.dataset.theme = theme;
    localStorage.setItem("theme", theme);
  }, [theme]);

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
      if (recordedUrl) URL.revokeObjectURL(recordedUrl);
    };
  }, [previewUrl, recordedUrl]);

  useEffect(() => {
    localStorage.setItem("history", JSON.stringify(history));
  }, [history]);

  const showNotification = (key, fallbackMessage) => {
    notification.open({
      message: t(key) || fallbackMessage,
      description: fallbackMessage,
      placement: "topRight",
    });
  };

  const handleFileSelection = (selectedFile) => {
    if (!selectedFile) return;
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    const nextUrl = URL.createObjectURL(selectedFile);
    setFile(selectedFile);
    setPreviewUrl(nextUrl);
    setRecordedBlob(null);
    setRecordedUrl("");
    showNotification("uploadSuccess", "Upload successful");
  };

  const uploadProps = {
    beforeUpload: (incomingFile) => {
      handleFileSelection(incomingFile);
      return false;
    },
    maxCount: 1,
    showUploadList: false,
    onDrop: () => setDragActive(false),
  };

  const predictEmotion = async (sourceFile = file) => {
    if (!sourceFile) {
      showNotification("predictionFailed", "Please select a WAV file first.");
      return;
    }

    const formData = new FormData();
    formData.append("file", sourceFile);

    setLoading(true);

    try {
      const response = await axios.post("http://127.0.0.1:8000/predict", formData);
      const nextEmotion = response.data.emotion || "neutral";
      const nextConfidence = Number(response.data.confidence || 0);
      setEmotion(nextEmotion);
      setConfidence(nextConfidence);
      setProbabilities(buildProbabilityData(nextEmotion, nextConfidence, response.data.probabilities));

      const nextHistoryItem = {
        id: Date.now(),
        emotion: nextEmotion,
        confidence: nextConfidence,
        fileName: sourceFile.name,
        timestamp: new Date().toISOString(),
      };
      setHistory((current) => [nextHistoryItem, ...current].slice(0, 12));
      showNotification("predictionSuccess", "Prediction completed");

      const speech = new SpeechSynthesisUtterance(
        `The predicted emotion is ${nextEmotion} with ${Math.round(nextConfidence)} percent confidence.`
      );
      window.speechSynthesis.cancel();
      window.speechSynthesis.speak(speech);
    } catch (error) {
      console.error(error);
      showNotification("networkError", "Prediction failed");
    } finally {
      setLoading(false);
    }
  };

  const handlePlayPause = async () => {
    const audio = audioRef.current;
    if (!audio) return;
    if (audio.paused) {
      await audio.play();
      setIsPlaying(true);
    } else {
      audio.pause();
      setIsPlaying(false);
    }
  };

  const handleReplay = () => {
    const audio = audioRef.current;
    if (!audio) return;
    audio.currentTime = 0;
    audio.play();
    setIsPlaying(true);
  };

  const handleSeek = (event) => {
    const audio = audioRef.current;
    if (!audio) return;
    const nextTime = Number(event.target.value);
    audio.currentTime = nextTime;
    setCurrentTime(nextTime);
  };

  const handleVolume = (event) => {
    const audio = audioRef.current;
    const nextVolume = Number(event.target.value);
    setVolume(nextVolume);
    if (audio) audio.volume = nextVolume;
  };

  const startRecording = async () => {
    if (!navigator.mediaDevices || !window.MediaRecorder) {
      showNotification("networkError", "Recording is not supported in this browser.");
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      const chunks = [];
      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) chunks.push(event.data);
      };
      recorder.onstop = () => {
        const blob = new Blob(chunks, { type: recorder.mimeType || "audio/webm" });
        const nextUrl = URL.createObjectURL(blob);
        setRecordedBlob(blob);
        setRecordedUrl(nextUrl);
        const nextFile = new File([blob], "recorded-audio.webm", { type: blob.type });
        handleFileSelection(nextFile);
        stream.getTracks().forEach((track) => track.stop());
      };
      setRecordedChunks(chunks);
      setMediaRecorder(recorder);
      recorder.start();
      setIsRecording(true);
    } catch (error) {
      console.error(error);
      showNotification("networkError", "Unable to access the microphone.");
    }
  };

  const stopRecording = () => {
    if (mediaRecorder && mediaRecorder.state !== "inactive") {
      mediaRecorder.stop();
      setIsRecording(false);
      setMediaRecorder(null);
    }
  };

  const clearHistory = () => {
    setHistory([]);
  };

  const downloadReport = () => {
    if (!emotion) return;
    const doc = new jsPDF();
    doc.setFontSize(20);
    doc.text(t("reportTitle"), 14, 20);
    doc.setFontSize(11);
    doc.text(t("reportSubtitle"), 14, 30);
    doc.text(`Filename: ${file?.name || "recorded-audio.webm"}`, 14, 45);
    doc.text(`Prediction: ${emotion}`, 14, 55);
    doc.text(`Confidence: ${confidence.toFixed(1)}%`, 14, 65);
    doc.text(`Date: ${new Date().toLocaleString()}`, 14, 75);
    doc.save(`${emotion}-report.pdf`);
  };

  const filteredHistory = history.filter((item) =>
    `${item.emotion} ${item.fileName}`.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div className={`app-shell ${theme}`}>
      <div className="background-glow glow-one" />
      <div className="background-glow glow-two" />
      <div className="background-glow glow-three" />
      
      <div className="app-container">
        {/* Header */}
        <header className="top-bar">
          <div className="header-content">
            <div className="logo-section">
              <div className="logo-icon">🎤</div>
              <div>
                <Title level={2} className="hero-title">
                  {t("title")}
                </Title>
                <Paragraph className="hero-subtitle">{t("subtitle")}</Paragraph>
              </div>
            </div>
          </div>
          <Space className="top-actions" size="large">
            <Select
              value={i18n.language}
              onChange={(value) => {
                i18n.changeLanguage(value);
                localStorage.setItem("language", value);
              }}
              options={[
                { value: "en", label: "English" },
                { value: "te", label: "తెలుగు" },
                { value: "hi", label: "हिंदी" },
              ]}
              className="lang-select"
              aria-label={t("language")}
            />
            <Button
              className="theme-toggle"
              onClick={() => setTheme(theme === "light" ? "dark" : "light")}
              icon={theme === "light" ? <MoonOutlined /> : <SunOutlined />}
              aria-label={t("theme")}
            />
          </Space>
        </header>

        {/* Main Content Grid */}
        <div className="main-grid">
          {/* Left Section - Upload & Player */}
          <div className="left-section">
            <Card className="panel upload-panel" bordered={false}>
              <div className="panel-header">📤 Upload Audio</div>
              <Upload.Dragger {...uploadProps} className={dragActive ? "dragging" : ""}>
                <p className="upload-icon-large">⬆️</p>
                <Title level={4}>{t("dragDrop")}</Title>
                <Paragraph>{t("browse")}</Paragraph>
                <Button type="primary" icon={<UploadOutlined />} size="large" className="upload-btn">
                  {t("chooseFile")}
                </Button>
              </Upload.Dragger>

              {file && (
                <div className="file-summary">
                  <div className="file-pill animate-slide-in">
                    <AudioOutlined />
                    <div>
                      <Text strong>{file.name}</Text>
                      <div className="file-size">{formatFileSize(file.size)}</div>
                    </div>
                  </div>
                </div>
              )}

              {previewUrl && (
                <div className="audio-player-card">
                  <div className="player-header-new">
                    <span className="player-icon">▶️</span>
                    <Text strong>Audio Playback</Text>
                  </div>
                  <audio
                    ref={audioRef}
                    src={previewUrl}
                    onLoadedMetadata={() => setDuration(audioRef.current?.duration || 0)}
                    onTimeUpdate={() => setCurrentTime(audioRef.current?.currentTime || 0)}
                    onEnded={() => setIsPlaying(false)}
                    onPlay={() => setIsPlaying(true)}
                    onPause={() => setIsPlaying(false)}
                  />
                  <div className="player-buttons">
                    <Button
                      shape="circle"
                      size="large"
                      icon={isPlaying ? <PauseCircleOutlined /> : <PlayCircleOutlined />}
                      onClick={handlePlayPause}
                      className="play-btn"
                      aria-label={t("play")}
                    />
                    <Button
                      shape="circle"
                      size="large"
                      icon={<ReloadOutlined />}
                      onClick={handleReplay}
                      className="replay-btn"
                      aria-label={t("replay")}
                    />
                  </div>
                  <div className="seek-container">
                    <input
                      type="range"
                      min="0"
                      max={duration || 100}
                      value={currentTime}
                      onChange={handleSeek}
                      className="seek-bar"
                      aria-label="Seek"
                    />
                    <div className="player-meta">
                      <span className="time-badge">{Math.floor(currentTime)}s</span>
                      <span className="time-badge">{Math.floor(duration)}s</span>
                    </div>
                  </div>
                  <div className="volume-container">
                    <SoundOutlined className="volume-icon" />
                    <input
                      type="range"
                      min="0"
                      max="1"
                      step="0.1"
                      value={volume}
                      onChange={handleVolume}
                      className="volume-bar"
                      aria-label={t("volume")}
                    />
                  </div>
                  <Button
                    type="primary"
                    size="large"
                    onClick={() => predictEmotion(file)}
                    loading={loading}
                    block
                    className="predict-btn"
                    style={{ marginTop: "12px" }}
                  >
                    {t("predict")} 🎯
                  </Button>
                </div>
              )}

              <div className="recording-card">
                <div className="recording-header">🎙️ Record Audio</div>
                <Space wrap className="recording-buttons">
                  <Button
                    icon={<VideoCameraOutlined />}
                    onClick={startRecording}
                    disabled={isRecording}
                    className="record-btn"
                  >
                    {t("start")}
                  </Button>
                  <Button onClick={stopRecording} disabled={!isRecording} className="stop-btn" danger>
                    {t("stop")}
                  </Button>
                </Space>
              </div>
            </Card>
          </div>

          {/* Right Section - Prediction & Results */}
          <div className="right-section">
            {loading ? (
              <div className="prediction-loading">
                <div className="loader-orb" />
                <Title level={3}>{t("processing")}</Title>
                <Text type="secondary">Analyzing your emotion...</Text>
              </div>
            ) : emotion ? (
              <>
                {/* Main Emotion Card */}
                <div className="main-emotion-display">
                  <div className="emotion-card-main" style={{ borderLeftColor: selectedProfile.color }}>
                    <div className="emotion-visual">
                      <div className="emotion-circle" style={{ background: `${selectedProfile.color}15` }}>
                        <span className="emotion-emoji animate-bounce">{selectedProfile.emoji}</span>
                      </div>
                      <div className="emotion-text">
                        <div className="emotion-name">{emotion}</div>
                        <div className="emotion-description">{selectedProfile.description}</div>
                      </div>
                    </div>
                    <div className="confidence-display">
                      <Progress
                        type="circle"
                        percent={Number(confidence.toFixed(1))}
                        size={100}
                        strokeColor={selectedProfile.color}
                        trailColor="rgba(148, 163, 184, 0.15)"
                        format={() => `${confidence.toFixed(0)}%`}
                      />
                    </div>
                  </div>
                  <div className="confidence-badge-row">
                    <span className={`confidence-indicator ${confidenceLevel.toLowerCase()}`}>
                      {confidenceLevel} Confidence
                    </span>
                    <Button type="primary" onClick={() => predictEmotion(file)} loading={loading} size="large" className="predict-btn">
                      {t("predict")} 🎯
                    </Button>
                    {emotion && (
                      <Button icon={<DownloadOutlined />} onClick={downloadReport}>
                        {t("download")}
                      </Button>
                    )}
                  </div>
                </div>

                {/* Probability Breakdown */}
                <div className="probability-breakdown">
                  <div className="prob-header">📊 Emotion Probabilities</div>
                  <div className="probability-grid">
                    {probabilities.slice(0, 8).map((entry) => {
                      const normalized = normalizeEmotion(entry.emotion);
                      const profile = emotionProfiles[normalized] || emotionProfiles.neutral;
                      const isHighlighted = normalizeEmotion(emotion) === normalized;
                      return (
                        <div
                          key={`${entry.emotion}-${entry.confidence}`}
                          className={`prob-item ${isHighlighted ? "highlighted" : ""}`}
                          style={isHighlighted ? { borderColor: profile.color } : {}}
                        >
                          <div className="prob-emoji">{profile.emoji}</div>
                          <div className="prob-name">{String(entry.emotion).slice(0, 8)}</div>
                          <div className="prob-bar-small">
                            <div
                              className="prob-fill"
                              style={{
                                width: `${Math.max(entry.confidence, 10)}%`,
                                background: profile.color,
                              }}
                            />
                          </div>
                          <div className="prob-percent">{Number(entry.confidence || 0).toFixed(0)}%</div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </>
            ) : (
              <div className="empty-prompt">
                <div className="empty-icon">🎵</div>
                <Title level={3}>Ready to Analyze</Title>
                <Paragraph>Upload an audio file and click "Predict" to discover the emotion in the speech</Paragraph>
              </div>
            )}
          </div>
        </div>

        {/* History Section */}
        <div className="history-section">
          <Card className="panel history-panel" bordered={false}>
            <div className="history-header">
              <div>
                <Title level={4}>📜 Prediction History</Title>
              </div>
              <Space>
                <Input.Search
                  placeholder={t("search")}
                  value={search}
                  onChange={(event) => setSearch(event.target.value)}
                  className="history-search"
                />
                <Button onClick={clearHistory} danger>
                  {t("clear")}
                </Button>
              </Space>
            </div>
            <div className="history-list-container">
              {filteredHistory.length > 0 ? (
                <div className="history-items">
                  {filteredHistory.map((item) => (
                    <div key={item.id} className="history-item">
                      <div className="history-item-content">
                        <div className="history-emotion">
                          <span className="history-emoji">{emotionProfiles[normalizeEmotion(item.emotion)]?.emoji || "😐"}</span>
                          <div>
                            <div className="history-emotion-name">{item.emotion}</div>
                            <div className="history-confidence">{item.confidence.toFixed(1)}% confidence</div>
                          </div>
                        </div>
                        <div className="history-file">
                          <div className="history-filename">{item.fileName}</div>
                          <div className="history-timestamp">{new Date(item.timestamp).toLocaleString()}</div>
                        </div>
                      </div>
                      <Button
                        icon={<DeleteOutlined />}
                        onClick={() => setHistory((current) => current.filter((entry) => entry.id !== item.id))}
                        danger
                        className="delete-btn"
                      />
                    </div>
                  ))}
                </div>
              ) : (
                <div className="empty-history">No predictions yet • Start by uploading an audio file</div>
              )}
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}

export default App;