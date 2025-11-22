import { useCallback, useEffect, useRef, useState } from 'react';
import './App.css';

const WS_ENDPOINT = '/ws/audio_filter';
const BUFFER_SIZE = 4096;
const INT16_MAX = 0x7fff;

const getWebSocketURL = () => {
  const host = window.location.hostname || 'localhost';
  return `ws://${host}:8000${WS_ENDPOINT}`;
};

const getAudioContextClass = () => window.AudioContext || window.webkitAudioContext;

const decodePCM16 = (arrayBuffer) => {
  if (!arrayBuffer || arrayBuffer.byteLength === 0) {
    return new Float32Array(0);
  }

  const int16Samples = new Int16Array(arrayBuffer);
  const floatSamples = new Float32Array(int16Samples.length);

  for (let i = 0; i < int16Samples.length; i += 1) {
    floatSamples[i] = int16Samples[i] / INT16_MAX;
  }

  return floatSamples;
};

function App() {
  const [isStreaming, setIsStreaming] = useState(false);
  const [statusMessage, setStatusMessage] = useState('Disconnected');
  const [errorMessage, setErrorMessage] = useState('');

  const wsRef = useRef(null);
  const mediaStreamRef = useRef(null);
  const captureContextRef = useRef(null);
  const sourceNodeRef = useRef(null);
  const processorRef = useRef(null);
  const silentGainRef = useRef(null);
  const playbackContextRef = useRef(null);
  const sampleRateRef = useRef(null);
  const playbackStartTimeRef = useRef(0);

  const cleanupCaptureNodes = useCallback(() => {
    if (processorRef.current) {
      processorRef.current.disconnect();
      processorRef.current.onaudioprocess = null;
      processorRef.current = null;
    }

    if (sourceNodeRef.current) {
      sourceNodeRef.current.disconnect();
      sourceNodeRef.current = null;
    }

    if (silentGainRef.current) {
      silentGainRef.current.disconnect();
      silentGainRef.current = null;
    }

    if (captureContextRef.current) {
      captureContextRef.current.close();
      captureContextRef.current = null;
    }

    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((track) => track.stop());
      mediaStreamRef.current = null;
    }
  }, []);

  const cleanupPlayback = useCallback(() => {
    if (playbackContextRef.current) {
      playbackContextRef.current.close();
      playbackContextRef.current = null;
    }
    playbackStartTimeRef.current = 0;
  }, []);

  const stopStreaming = useCallback(() => {
    const ws = wsRef.current;
    wsRef.current = null;

    cleanupCaptureNodes();
    cleanupPlayback();

    if (ws && ws.readyState !== WebSocket.CLOSED && ws.readyState !== WebSocket.CLOSING) {
      ws.close();
    }

    setIsStreaming(false);
    setStatusMessage('Disconnected');
  }, [cleanupCaptureNodes, cleanupPlayback]);

  const ensurePlaybackContext = useCallback(
    async (preferredSampleRate) => {
      if (!playbackContextRef.current) {
        const AudioContextClass = getAudioContextClass();
        if (!AudioContextClass) {
          throw new Error('Web Audio API not supported in this browser.');
        }

        try {
          playbackContextRef.current = preferredSampleRate
            ? new AudioContextClass({ sampleRate: preferredSampleRate })
            : new AudioContextClass();
        } catch (error) {
          playbackContextRef.current = new AudioContextClass();
        }

        playbackStartTimeRef.current = playbackContextRef.current.currentTime;
      }

      const context = playbackContextRef.current;
      if (context.state === 'suspended') {
        await context.resume();
      }

      return context;
    },
    []
  );

  const startStreaming = useCallback(async () => {
    if (isStreaming) {
      return;
    }

    setErrorMessage('');
    setStatusMessage('Requesting microphone…');

    try {
      const AudioContextClass = getAudioContextClass();
      if (!AudioContextClass) {
        throw new Error('Web Audio API not supported in this browser.');
      }

      const mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true },
      });
      mediaStreamRef.current = mediaStream;

      const captureContext = new AudioContextClass();
      captureContextRef.current = captureContext;
      await captureContext.resume();

      sampleRateRef.current = captureContext.sampleRate;
      await ensurePlaybackContext(sampleRateRef.current);

      const sourceNode = captureContext.createMediaStreamSource(mediaStream);
      const processorNode = captureContext.createScriptProcessor(BUFFER_SIZE, 1, 1);
      sourceNodeRef.current = sourceNode;
      processorRef.current = processorNode;

      const silentGain = captureContext.createGain();
      silentGain.gain.value = 0;
      silentGainRef.current = silentGain;

      sourceNode.connect(processorNode);
      processorNode.connect(silentGain);
      silentGain.connect(captureContext.destination);

      setStatusMessage('Connecting to server…');

      const ws = new WebSocket(getWebSocketURL());
      ws.binaryType = 'arraybuffer';
      wsRef.current = ws;

      ws.onopen = () => {
        setIsStreaming(true);
        setStatusMessage('Streaming');
        if (sampleRateRef.current) {
          ws.send(
            JSON.stringify({
              type: 'config',
              sampleRate: sampleRateRef.current,
            })
          );
        }
      };

      ws.onmessage = async (event) => {
        if (!playbackContextRef.current) {
          return;
        }

        let arrayBuffer;
        if (event.data instanceof ArrayBuffer) {
          arrayBuffer = event.data;
        } else if (event.data instanceof Blob) {
          arrayBuffer = await event.data.arrayBuffer();
        } else {
          return;
        }

        const floatData = decodePCM16(arrayBuffer);
        if (!floatData.length) {
          return;
        }

        const playbackContext = await ensurePlaybackContext(sampleRateRef.current);
        const buffer = playbackContext.createBuffer(
          1,
          floatData.length,
          sampleRateRef.current || playbackContext.sampleRate
        );
        buffer.copyToChannel(floatData, 0);

        const source = playbackContext.createBufferSource();
        source.buffer = buffer;
        source.connect(playbackContext.destination);

        const now = playbackContext.currentTime;
        const startAt = Math.max(playbackStartTimeRef.current, now + 0.01);
        source.start(startAt);
        playbackStartTimeRef.current = startAt + buffer.duration;
      };

      ws.onerror = (event) => {
        console.error('WebSocket error:', event);
        setErrorMessage('Connection error. Check the backend service.');
      };

      ws.onclose = () => {
        stopStreaming();
      };

      processorNode.onaudioprocess = ({ inputBuffer }) => {
        if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
          return;
        }

        const channelData = inputBuffer.getChannelData(0);
        const payload = new Float32Array(channelData);
        wsRef.current.send(payload.buffer.slice(0));
      };
    } catch (error) {
      console.error(error);
      setErrorMessage(error.message || 'Failed to start streaming.');
      stopStreaming();
    }
  }, [ensurePlaybackContext, isStreaming, stopStreaming]);

  useEffect(() => {
    return () => {
      stopStreaming();
    };
  }, [stopStreaming]);

  return (
    <div className="App scribe-app">
      <h1>Realtime Audio Filter</h1>
      <p>Status: {statusMessage}</p>
      {errorMessage && <p className="error">{errorMessage}</p>}
      <div className="controls">
        <button onClick={startStreaming} disabled={isStreaming}>
          Start Streaming
        </button>
        <button onClick={stopStreaming} disabled={!isStreaming}>
          Stop
        </button>
      </div>
    </div>
  );
}

export default App;
