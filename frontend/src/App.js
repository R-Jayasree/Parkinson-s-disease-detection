import React, { useState, useRef } from "react";
import { useTranslation } from 'react-i18next';
import "./App.css";
import "./i18n"; 

function App() {
  const { t, i18n } = useTranslation();
  const [activeTab, setActiveTab] = useState("predictions");
  const [audioFile, setAudioFile] = useState(null);
  const [audioURL, setAudioURL] = useState("");
  const [isRecording, setIsRecording] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState("");
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const canvasRef = useRef(null);
  const animationRef = useRef(null);
  const audioContextRef = useRef(null);
  const analyserRef = useRef(null);
  const dataArrayRef = useRef(null);

  const changeLanguage = (lang) => {
    i18n.changeLanguage(lang);
  };
const Navbar = () => (
  <nav className="navbar">
    <div className="navbar-left">
      <img src="https://cdn-icons-png.flaticon.com/512/5996/5996521.png" alt="App Icon" className="app-icon" />
      <h2>{t('navbar.title')}</h2>
    </div>
    <div className="navbar-tabs">
      <button 
        className={activeTab === "predictions" ? "active" : ""}
        onClick={() => setActiveTab("predictions")}
      >
        {t('navbar.predictions')}
      </button>
      
      <div className="language-selector">
        <select 
          value={i18n.language} 
          onChange={(e) => changeLanguage(e.target.value)}
          className="language-dropdown"
        >
          <option value="en">🌐 English</option>
          <option value="hi">🌐 हिंदी</option>
          <option value="ta">🌐 தமிழ்</option>
          <option value="te">🌐 తెలుగు</option>
          <option value="bn">🌐 বাংলা</option>
          <option value="mr">🌐 मराठी</option>
          <option value="kn">🌐 ಕನ್ನಡ</option>
          <option value="ml">🌐 മലയാളം</option>
          <option value="gu">🌐 ગુજરાતી</option>
        </select>
      </div>
    </div>
  </nav>
);

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      setAudioFile(file);
      setAudioURL(URL.createObjectURL(file));
      setPrediction(null);
      setError("");
    }
  };

  const startRecording = async () => {
    setError("");
    setPrediction(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];
      mediaRecorder.start();
      setIsRecording(true);
      
      mediaRecorder.ondataavailable = (event) => {
        audioChunksRef.current.push(event.data);
      };

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: "audio/webm" });
        const wavBlob = await convertToWav(audioBlob);
        const wavFile = new File([wavBlob], "recording.wav", { type: "audio/wav" });
        setAudioFile(wavFile);
        setAudioURL(URL.createObjectURL(wavBlob));
      };
    } catch (err) {
      setError(t('errors.microphoneAccess'));
    }
  };

  // Stop recording
  const stopRecording = () => {
    if (mediaRecorderRef.current) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  // Waveform setup
  const setupWaveform = (stream) => {
    audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)();
    const source = audioContextRef.current.createMediaStreamSource(stream);
    analyserRef.current = audioContextRef.current.createAnalyser();
    analyserRef.current.fftSize = 256;

    const bufferLength = analyserRef.current.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    dataArrayRef.current = dataArray;

    source.connect(analyserRef.current);
    drawWaveform();
  };

  const drawWaveform = () => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    const analyser = analyserRef.current;
    const dataArray = dataArrayRef.current;

    const draw = () => {
      animationRef.current = requestAnimationFrame(draw);
      analyser.getByteTimeDomainData(dataArray);
      ctx.fillStyle = "#f5f7fb";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.lineWidth = 2;
      ctx.strokeStyle = "#2b4c7e";
      ctx.beginPath();
      const sliceWidth = (canvas.width * 1.0) / dataArray.length;
      let x = 0;
      for (let i = 0; i < dataArray.length; i++) {
        const v = dataArray[i] / 128.0;
        const y = (v * canvas.height) / 2;
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        x += sliceWidth;
      }
      ctx.lineTo(canvas.width, canvas.height / 2);
      ctx.stroke();
    };
    draw();
  };

  // Convert to WAV
  const convertToWav = async (webmBlob) => {
    const arrayBuffer = await webmBlob.arrayBuffer();
    const audioContext = new AudioContext();
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
    const wavBuffer = audioBufferToWav(audioBuffer);
    return new Blob([wavBuffer], { type: "audio/wav" });
  };

  const audioBufferToWav = (buffer) => {
    const numOfChan = buffer.numberOfChannels;
    const length = buffer.length * numOfChan * 2 + 44;
    const bufferOut = new ArrayBuffer(length);
    const view = new DataView(bufferOut);
    let offset = 0;

    const writeString = (s) => {
      for (let i = 0; i < s.length; i++) {
        view.setUint8(offset + i, s.charCodeAt(i));
      }
      offset += s.length;
    };

    const channels = [];
    for (let i = 0; i < numOfChan; i++) channels.push(buffer.getChannelData(i));

    writeString("RIFF");
    view.setUint32(offset, 36 + buffer.length * numOfChan * 2, true);
    offset += 4;
    writeString("WAVE");
    writeString("fmt ");
    view.setUint32(offset, 16, true);
    offset += 4;
    view.setUint16(offset, 1, true);
    offset += 2;
    view.setUint16(offset, numOfChan, true);
    offset += 2;
    view.setUint32(offset, buffer.sampleRate, true);
    offset += 4;
    view.setUint32(offset, buffer.sampleRate * numOfChan * 2, true);
    offset += 4;
    view.setUint16(offset, numOfChan * 2, true);
    offset += 2;
    view.setUint16(offset, 16, true);
    offset += 2;
    writeString("data");
    view.setUint32(offset, buffer.length * numOfChan * 2, true);
    offset += 4;

    for (let i = 0; i < buffer.length; i++) {
      for (let ch = 0; ch < numOfChan; ch++) {
        const sample = Math.max(-1, Math.min(1, channels[ch][i]));
        view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
        offset += 2;
      }
    }

    return bufferOut;
  };

  const handleSubmit = async () => {
    if (!audioFile) {
      setError(t('errors.noFile'));
      return;
    }

    setError("");
    setPrediction(null);
    const formData = new FormData();
    formData.append("file", audioFile);

    try {
      const response = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        body: formData,
      });
      if (!response.ok) throw new Error("Failed to get prediction");
      const data = await response.json();
      setPrediction(data);
    } catch (err) {
      setError(t('errors.predictionFailed'));
    }
  };

  return (
    <>
      <Navbar />
      <div className="app-container">
        {activeTab === "predictions" && (
          <div className="predict-container">
            <h3>{t('hero.subtitle')}</h3>
            <h2>{t('upload.title')}</h2>

            <div className="action-buttons">
              <label className="upload-btn">
                {t('upload.uploadBtn')}
                <input type="file" accept="audio/*" onChange={handleFileChange} hidden />
              </label>

              {!isRecording ? (
                <button className="record-btn" onClick={startRecording}>
                  {t('upload.startRecording')}
                </button>
              ) : (
                <button className="stop-btn" onClick={stopRecording}>
                  {t('upload.stopRecording')}
                </button>
              )}
            </div>

            {isRecording && (
              <div className="wave-container">
                <canvas ref={canvasRef} width="500" height="100"></canvas>
              </div>
            )}

            {audioURL && (
              <div className="audio-preview">
                <h3>{t('upload.preview')}</h3>
                <audio controls src={audioURL}></audio>
              </div>
            )}

            <button className="predict-btn" onClick={handleSubmit}>
              {t('upload.predictBtn')}
            </button>

            {error && <p className="error">{error}</p>}

            {prediction && (
              <div className="result-card">
                <h2>{t('results.title')}</h2>
                <p>
                  <strong>{t('results.status')}:</strong> {prediction.prediction}
                </p>
                <p>
                  <strong>{t('results.riskScore')}:</strong> {prediction.risk_score.toFixed(3)}
                </p>
                <h3>{t('results.suggestions')}</h3>
                <ul>
                  {prediction.prediction === "Parkinson’s Detected" ? (
                    t('results.suggestionsDetected', { returnObjects: true }).map((suggestion, idx) => (
                      <li key={idx}>{suggestion}</li>
                    ))
                  ) : (
                    t('results.suggestionsHealthy', { returnObjects: true }).map((suggestion, idx) => (
                      <li key={idx}>{suggestion}</li>
                    ))
                  )}
                </ul>
              </div>
            )}
          </div>
        )}
      </div>
    </>
  );
}

export default App;
