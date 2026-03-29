"use client";
import React, { useRef, useEffect, useState } from "react";
import Webcam from "react-webcam";

// Your custom English-to-Thai dictionary
const thaiDictionary = {
  "hello": "สวัสดี",
  "me": "ฉัน",
  "you": "คุณ",
  "bye": "ลาก่อน",
  "sick": "ป่วย",
  "yes": "ใช่",
  "no": "ไม่",
  "help": "ช่วย",
  "eat": "กิน",
  "goodluck": "โชคดี"
  // Add the rest of the exact words your AI was trained on here!
};

export default function Home() {
  const webcamRef = useRef(null);
  const wsRef = useRef(null);

  const [status, setStatus] = useState("Disconnected");
  const [currentWord, setCurrentWord] = useState("Waiting...");
  const [sentence, setSentence] = useState([]);
  const [activeModel, setActiveModel] = useState("default");

  // A reference to remember the last word spoken (Crucial for Teachable Machine anti-spam!)
  const lastSpokenWordRef = useRef("");

  useEffect(() => {
    // Automatically use the correct WebSocket address (ws:// for localhost, wss:// for Cloudflare)
    const protocol = window.location.protocol === "https:" ? "wss://" : "ws://";
    const ws = new WebSocket(protocol + window.location.host + "/ws/translate/");
    wsRef.current = ws; // Save the connection to our reference!
    let captureInterval;

    ws.onopen = () => {
      setStatus("Connected");
      captureInterval = setInterval(() => {
        if (webcamRef.current) {
          const imageSrc = webcamRef.current.getScreenshot();
          if (imageSrc && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ frame: imageSrc }));
          }
        }
      }, 100); // 10 FPS Sweet Spot
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.prediction) {
        // 1. Clean the incoming English word (make it lowercase just in case)
        const englishWord = data.prediction.toLowerCase();

        // 2. Look it up in the dictionary. If it"s missing, just use the English word.
        const thaiWord = thaiDictionary[englishWord] || englishWord;

        // FRONTEND ANTI-SPAM: Only trigger if the AI says a NEW word!
        if (thaiWord !== lastSpokenWordRef.current) {
          lastSpokenWordRef.current = thaiWord; // Update the memory

          // 3. Update the UI with the Thai word!
          setCurrentWord(thaiWord);
          setSentence((prev) => [...prev, thaiWord]);

          // 4. Speak it in Thai!
          const utterance = new SpeechSynthesisUtterance(thaiWord);
          utterance.lang = "th-TH"; // <-- This is the magic line that forces the Thai voice!
          utterance.rate = 0.9;     // (Optional) Slows the robot down just a tiny bit so it sounds more natural
          window.speechSynthesis.speak(utterance);
        }
      }
    };

    ws.onclose = () => setStatus("Disconnected");

    return () => {
      clearInterval(captureInterval);
      ws.close();
    };
  }, []);

  // --- INTERACTIVE FEATURES ---

  const handleModelChange = (e) => {
    const newModel = e.target.value;
    setActiveModel(newModel);

    // Tell the Django Router to switch engines!
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        command: "switch_model",
        model: newModel
      }));
    }

    // Reset the frontend memory when switching models
    lastSpokenWordRef.current = "";
    setCurrentWord("Waiting...");
  };

  const clearSentence = () => {
    setSentence([]);
    setCurrentWord("Waiting...");
    lastSpokenWordRef.current = ""; // Clear anti-spam memory so you can repeat a word
  };

  const speakText = () => {
    if (sentence.length > 0) {
      // Use the browser"s built-in robot voice to read the array of words
      const utterance = new SpeechSynthesisUtterance(sentence.join(" "));
      utterance.lang = "th-TH"; // Force Thai here too!
      window.speechSynthesis.speak(utterance);
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white font-sans flex flex-col items-center py-10">

      <h1 className="text-4xl font-bold mb-2 tracking-wide text-blue-400">Sign Language Translator</h1>
      <div className="flex items-center space-x-3 mb-8">
        <div className={`h-3 w-3 rounded-full ${status === "Connected" ? "bg-green-500 animate-pulse" : "bg-red-500"}`}></div>
        <p className="text-gray-300 font-medium">System Status: <span className={status === "Connected" ? "text-green-400" : "text-red-400"}>{status}</span></p>
      </div>

      <div className="flex flex-col md:flex-row gap-8 w-full max-w-6xl px-4">

        {/* Left Side: Camera & AI Controls */}
        <div className="flex-1 flex flex-col gap-4">

          {/* THE NEW DROPDOWN MENU */}
          <div className="bg-gray-800 p-4 rounded-2xl shadow-lg border border-gray-700">
            <label className="block text-gray-400 text-sm font-bold mb-2">
              🧠 Select AI Engine
            </label>
            <select 
              value={activeModel} 
              onChange={handleModelChange}
              className="w-full bg-gray-900 text-white border border-gray-600 rounded-lg py-2 px-3 focus:outline-none focus:border-blue-500 transition"
            >
              <optgroup label="LSTM Sequence Models (Dynamic Signs)">
                <option value="default">V1: Default (Face Included)</option>
                <option value="less_face">V1: Focused (Hands & Pose Only)</option>
                <option value="default_v2">V2: Default (Face Included)</option>
                <option value="less_face_v2">V2: Focused (Hands & Pose Only)</option>
              </optgroup>
              <optgroup label="CNN Image Models (Static Signs)">
                <option value="teachable_machine">Teachable Machine (Raw Camera Pixels)</option>
              </optgroup>
            </select>
          </div>

          <div className="bg-gray-800 p-4 rounded-2xl shadow-2xl border border-gray-700">
            <Webcam
              ref={webcamRef}
              className="w-full rounded-xl transform scale-x-[-1]"
            />
            <div className="mt-4 text-center">
              <p className="text-gray-400 text-sm uppercase tracking-widest">Latest Sign</p>
              <p className="text-3xl font-bold text-yellow-400">{currentWord}</p>
            </div>
          </div>
        </div>

        {/* Right Side: Transcript */}
        <div className="flex-1 flex flex-col bg-gray-800 p-6 rounded-2xl shadow-2xl border border-gray-700">
          <h2 className="text-xl font-semibold mb-4 text-gray-200 border-b border-gray-600 pb-2">Live Transcript</h2>

          <div className="flex-1 bg-gray-900 rounded-xl p-4 mb-6 border border-gray-700 min-h-[200px]">
            <p className="text-2xl leading-relaxed">
              {sentence.length > 0 ? sentence.join(" ") : <span className="text-gray-600 italic">Start signing to build a sentence...</span>}
            </p>
          </div>

          <div className="flex gap-4">
            <button
              onClick={speakText}
              className="flex-1 bg-blue-600 hover:bg-blue-500 text-white font-bold py-3 px-4 rounded-xl transition duration-200 shadow-lg"
            >
              🔊 Read Aloud
            </button>
            <button
              onClick={clearSentence}
              className="flex-1 bg-red-600 hover:bg-red-500 text-white font-bold py-3 px-4 rounded-xl transition duration-200 shadow-lg"
            >
              🗑️ Clear Text
            </button>
          </div>
        </div>

      </div>
    </div>
  );
}