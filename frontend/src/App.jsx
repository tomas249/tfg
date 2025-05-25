import React, { useState, useEffect, useRef } from "react";
import axios from "axios";
import sampleData from "./assets/sample_data.json";
import { NEURON_FEATURES, PREDICTION_FEATURES } from "./constants.js";

const getApiBase = () => {
  if (window.location.hostname === "localhost") {
    return "http://localhost:4001";
  }
  return "https://api.tfg.tomastm.com";
};

function makeInputBlob(neuron) {
  return NEURON_FEATURES.reduce((acc, f) => {
    const v = neuron[f.label];
    if (f.map && v !== "" && v != null) {
      return { ...acc, ...f.map(v) };
    }
    return acc;
  }, {});
}

function App() {
  const [sessionToken, setSessionToken] = useState(
    localStorage.getItem("session-token") || "",
  );
  const [sessionInput, setSessionInput] = useState("");
  const [neurons, setNeurons] = useState([
    Object.fromEntries(NEURON_FEATURES.map((f) => [f.label, ""])),
  ]);
  const [predictions, setPredictions] = useState([]);
  const [loadingPredictions, setLoadingPredictions] = useState(false);
  const [predicting, setPredicting] = useState(false);
  const [error, setError] = useState("");
  const predictionsIntervalRef = useRef(null);
  const [sampleIndex, setSampleIndex] = useState(0);

  function sampleToNeuronFormRow(sample) {
    let side = "";
    if (sample.side_center) side = "center";
    else if (sample.side_left) side = "left";
    else if (sample.side_right) side = "right";

    let flow = "";
    if (sample.flow_afferent) flow = "afferent";
    else if (sample.flow_efferent) flow = "efferent";
    else if (sample.flow_intrinsic) flow = "intrinsic";

    return {
      "Length (nm)": sample.length_nm?.toString() || "",
      "Size (nm)": sample.size_nm?.toString() || "",
      "Area (nm)": sample.area_nm?.toString() || "",
      x: sample.centroid_x?.toString() || "",
      y: sample.centroid_y?.toString() || "",
      z: sample.centroid_z?.toString() || "",
      Side: side,
      Flow: flow,
      "Input Neuropil": sample.input_neuropil || "",
      "Output Neuropil": sample.output_neuropil || "",
      "Input Synapses Count": sample.input_synapses_count?.toString() || "",
      "Output Synapses Count": sample.output_synapses_count?.toString() || "",
      "Input Partners Count": sample.input_partners_count?.toString() || "",
      "Output Partners Count": sample.output_partners_count?.toString() || "",
    };
  }

  function handleAddSample() {
    const sample = sampleData[sampleIndex];
    setNeurons([...neurons, sampleToNeuronFormRow(sample)]);
    setSampleIndex(sampleIndex + 1);
  }

  // Sync sessionToken to localStorage
  useEffect(() => {
    if (sessionToken) {
      localStorage.setItem("session-token", sessionToken);
    } else {
      localStorage.removeItem("session-token");
    }
  }, [sessionToken]);

  // On login, fetch predictions and start interval
  useEffect(() => {
    if (!sessionToken) return;
    fetchPredictions();

    // Poll predictions every 10 seconds
    predictionsIntervalRef.current = setInterval(fetchPredictions, 10000);
    return () => clearInterval(predictionsIntervalRef.current);
    // eslint-disable-next-line
  }, [sessionToken]);

  // --- API Calls ---
  async function fetchPredictions() {
    setLoadingPredictions(true);
    setError("");
    try {
      const res = await axios.get(
        `${getApiBase()}/sessions/${sessionToken}/predictions`,
      );
      setPredictions(res.data || []);
    } catch (e) {
      setPredictions([]);
      setError("Could not load predictions (invalid session token?)");
    }
    setLoadingPredictions(false);
  }

  async function handleSessionGenerate() {
    setError("");
    try {
      const res = await axios.post(`${getApiBase()}/sessions`);
      setSessionToken(res.data.token);
    } catch (e) {
      setError("Could not generate session.");
    }
  }

  async function handleSessionEnter() {
    if (!sessionInput.trim()) return;
    setSessionToken(sessionInput.trim());
    setSessionInput("");
    setError("");
  }

  function handleSessionExit() {
    setSessionToken("");
    setPredictions([]);
    setNeurons([Object.fromEntries(NEURON_FEATURES.map((f) => [f.label, ""]))]);
    setError("");
  }
  function handleRemoveNeuron(idx) {
    if (neurons.length === 1) return; // Prevent removing the last row
    setNeurons((prev) => prev.filter((_, i) => i !== idx));
  }
  function handleNeuronInput(rowIdx, field, value) {
    setNeurons((prev) =>
      prev.map((n, idx) => (idx === rowIdx ? { ...n, [field]: value } : n)),
    );
  }

  function handleAddNeuron() {
    setNeurons([
      ...neurons,
      Object.fromEntries(NEURON_FEATURES.map((f) => [f.label, ""])),
    ]);
  }

  async function handlePredict() {
    setPredicting(true);
    setError("");
    try {
      // Build API blob for each neuron
      const inputs = neurons.map(makeInputBlob);
      await axios.post(`${getApiBase()}/predictions`, {
        token: sessionToken,
        inputs,
      });
      await fetchPredictions();
      // Optionally clear the form after submit
      setNeurons([
        Object.fromEntries(NEURON_FEATURES.map((f) => [f.label, ""])),
      ]);
    } catch (e) {
      setError("Failed to submit predictions.");
    }
    setPredicting(false);
  }

  // --- RENDER ---
  return (
    <div className="min-h-screen bg-gray-50 flex flex-col items-center">
      {/* Header */}
      <header className="w-full flex items-center justify-center py-2 shadow bg-white mb-8">
        <img src="./uoc.png" alt="University Logo" className="max-h-28" />
        <div className="ml-28 flex-1">
          <h1 className="text-3xl font-bold">
            Sistema de clasificación de neuronas
          </h1>
          <div className="text-xl text-gray-600 font-medium">
            por Tomas Mirchev
          </div>
        </div>
      </header>

      {!sessionToken && (
        <main className="flex flex-1 flex-col items-center justify-center w-full">
          <div className="bg-white rounded-2xl shadow-xl p-8 flex flex-col items-center gap-6">
            {error && <div className="text-red-600">{error}</div>}
            <input
              type="text"
              placeholder="Paste session token"
              value={sessionInput}
              onChange={(e) => setSessionInput(e.target.value)}
              className="border p-2 rounded w-72 text-lg"
            />
            <div className="flex gap-4">
              <button
                onClick={handleSessionEnter}
                className="bg-blue-600 text-white px-6 py-2 rounded-xl hover:bg-blue-700 transition"
              >
                Enter Session
              </button>
              <button
                onClick={handleSessionGenerate}
                className="bg-green-600 text-white px-6 py-2 rounded-xl hover:bg-green-700 transition"
              >
                Generate New Session
              </button>
            </div>
          </div>
        </main>
      )}

      {sessionToken && (
        <main className="w-full max-w-8xl flex flex-col gap-8 px-4">
          {/* Session info */}
          <div className="flex items-center justify-between mt-4 mb-2">
            <div className="text-gray-700 text-lg font-semibold">
              Session:{" "}
              <span className="font-mono bg-gray-100 px-2 py-1 rounded">
                {sessionToken}
              </span>
            </div>
            <button
              onClick={handleSessionExit}
              className="text-red-600 border border-red-200 px-4 py-1 rounded-xl hover:bg-red-50 transition"
            >
              Exit Session
            </button>
          </div>

          {error && <div className="text-red-600 mb-2">{error}</div>}

          {/* Neuron Input Form */}
          <div className="bg-white p-6 rounded-2xl shadow-xl">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-xl font-bold">Neuron Data</h2>
              <div className="flex gap-2">
                <button
                  onClick={handleAddNeuron}
                  className="flex items-center gap-2 text-blue-700 border border-blue-300 px-3 py-1 rounded-xl hover:bg-blue-50 transition"
                  title="Add Neuron"
                >
                  <span className="text-xl leading-none">+</span>
                  <span className="sr-only">Add Neuron</span>
                </button>
                <button
                  onClick={handleAddSample}
                  className="flex items-center gap-2 text-green-700 border border-green-300 px-3 py-1 rounded-xl hover:bg-green-50 transition"
                  title="Add Sample Neuron"
                >
                  <span className="text-sm leading-none">
                    Add Sample Neuron
                  </span>
                  <span className="sr-only">Add Sample Neuron</span>
                </button>
              </div>
            </div>
            {/* Dynamic Table */}
            <div className="overflow-x-auto">
              <table className="w-full text-left border">
                <thead>
                  <tr>
                    {NEURON_FEATURES.map((feature) => (
                      <th
                        key={feature.label}
                        className="border px-3 py-2 bg-gray-100 text-nowrap"
                      >
                        {feature.label}
                      </th>
                    ))}
                    <th className="border px-3 py-2 bg-gray-100"></th>{" "}
                    {/* Remove column */}
                  </tr>
                </thead>
                <tbody>
                  {neurons.map((neuron, rowIdx) => (
                    <tr key={rowIdx} className="bg-white">
                      {NEURON_FEATURES.map((feature) => (
                        <td key={feature.label} className="border px-3 py-2">
                          {feature.type === "select" ? (
                            <select
                              className="w-full border-b px-1 py-1 focus:outline-none focus:border-blue-400"
                              value={neuron[feature.label]}
                              onChange={(e) =>
                                handleNeuronInput(
                                  rowIdx,
                                  feature.label,
                                  e.target.value,
                                )
                              }
                            >
                              <option value="">Select...</option>
                              {feature.options.map((opt) => (
                                <option key={opt.value} value={opt.value}>
                                  {opt.label}
                                </option>
                              ))}
                            </select>
                          ) : (
                            <input
                              type={feature.type}
                              className="w-full border-b px-1 py-1 focus:outline-none focus:border-blue-400"
                              value={neuron[feature.label]}
                              onChange={(e) =>
                                handleNeuronInput(
                                  rowIdx,
                                  feature.label,
                                  e.target.value,
                                )
                              }
                            />
                          )}
                        </td>
                      ))}
                      {/* Minus button */}
                      <td className="border px-3 py-2 text-center">
                        <button
                          type="button"
                          className="text-red-500 text-2xl font-bold hover:text-red-700 transition"
                          title="Remove row"
                          onClick={() => handleRemoveNeuron(rowIdx)}
                          disabled={neurons.length === 1} // Prevent removing the last row
                        >
                          –
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="flex justify-end mt-4">
              <button
                onClick={handlePredict}
                disabled={predicting}
                className={`bg-blue-600 text-white px-6 py-2 rounded-xl transition font-semibold ${
                  predicting
                    ? "opacity-50 cursor-not-allowed"
                    : "hover:bg-blue-700"
                }`}
              >
                {predicting ? "Predicting..." : "Predict"}
              </button>
            </div>
          </div>

          {/* Predictions Table */}
          <div className="bg-white p-6 rounded-2xl shadow-xl">
            <h2 className="text-xl font-bold mb-4 flex items-center">
              Predictions
              {loadingPredictions && (
                <span className="ml-3 text-blue-500 text-base animate-pulse">
                  Updating...
                </span>
              )}
            </h2>
            <div className="overflow-x-auto">
              <table className="w-full text-left border">
                <thead>
                  <tr>
                    {PREDICTION_FEATURES.map((f) => (
                      <th key={f.key} className="border px-3 py-2 bg-gray-100">
                        {f.label}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {predictions.length === 0 && (
                    <tr>
                      <td
                        colSpan={PREDICTION_FEATURES.length}
                        className="text-center py-6 text-gray-500"
                      >
                        No predictions yet
                      </td>
                    </tr>
                  )}
                  {predictions.map((pred, idx) => (
                    <tr key={pred.id || idx} className="bg-white">
                      {PREDICTION_FEATURES.map((f) => (
                        <td key={f.key} className="border px-3 py-2">
                          {pred.output && pred.output[f.key]
                            ? pred.output[f.key]
                            : pred[f.key] || "-"}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </main>
      )}
    </div>
  );
}

export default App;
