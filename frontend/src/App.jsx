import { useMemo, useState } from "react";

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

const defaultForm = {
  date: "2025-12-31",
  state: "TAS",
  suburb: "Hobart",
  postcode: "7000",
  bedrooms: 2,
  dwelling_type: "unit"
};

export default function App() {
  const [form, setForm] = useState(defaultForm);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);

  const previewJson = useMemo(() => JSON.stringify(form, null, 2), [form]);

  const updateField = (field) => (event) => {
    const value = field === "bedrooms" ? Number(event.target.value) : event.target.value;
    setForm((prev) => ({ ...prev, [field]: value }));
  };

  const submit = async (event) => {
    event.preventDefault();
    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form)
      });

      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload?.detail || `Request failed with status ${response.status}`);
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message || "Something went wrong.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page">
      <header className="hero">
        <div>
          <p className="eyebrow">MoveMate • Rent Intelligence</p>
          <h1>Predict weekly rent with confidence.</h1>
          <p className="subtitle">
            A fast, suburb-aware model that blends live categorical intelligence with lagged rent signals.
          </p>
          <div className="pill-row">
            <span>CatBoost</span>
            <span>Baseline median</span>
            <span>Time-split validation</span>
          </div>
        </div>
        <div className="stat-card">
          <div>
            <p className="stat-label">API</p>
            <p className="stat-value">{API_BASE}</p>
          </div>
          <div>
            <p className="stat-label">Endpoints</p>
            <p className="stat-value">/predict • /predict-batch</p>
          </div>
        </div>
      </header>

      <main className="grid">
        <section className="card form-card">
          <h2>Single prediction</h2>
          <p className="card-subtitle">Fill the dwelling details and get a rent estimate.</p>
          <form onSubmit={submit} className="form">
            <label>
              Date
              <input type="date" value={form.date} onChange={updateField("date")} required />
            </label>
            <label>
              State
              <input type="text" value={form.state} onChange={updateField("state")} required />
            </label>
            <label>
              Suburb
              <input type="text" value={form.suburb} onChange={updateField("suburb")} required />
            </label>
            <label>
              Postcode
              <input type="text" value={form.postcode} onChange={updateField("postcode")} required />
            </label>
            <label>
              Bedrooms
              <input type="number" min="0" value={form.bedrooms} onChange={updateField("bedrooms")} required />
            </label>
            <label>
              Dwelling type
              <select value={form.dwelling_type} onChange={updateField("dwelling_type")} required>
                <option value="house">house</option>
                <option value="unit">unit</option>
                <option value="apartment">apartment</option>
              </select>
            </label>
            <button type="submit" disabled={loading}>
              {loading ? "Predicting..." : "Get prediction"}
            </button>
          </form>
        </section>

        <section className="card result-card">
          <h2>Prediction output</h2>
          <p className="card-subtitle">Model estimates in AUD/week.</p>
          {error && <div className="alert">{error}</div>}
          {!error && !result && <div className="empty">Submit a request to see results.</div>}
          {result && (
            <div className="results">
              <div>
                <p className="label">CatBoost</p>
                <p className="value">${result.prediction_catboost.toFixed(0)}</p>
              </div>
              <div>
                <p className="label">Baseline</p>
                <p className="value">${result.prediction_baseline.toFixed(0)}</p>
              </div>
            </div>
          )}
          <div className="json-preview">
            <p>Request payload</p>
            <pre>{previewJson}</pre>
          </div>
        </section>

        <section className="card info-card">
          <h2>Batch ready</h2>
          <p className="card-subtitle">
            Use /predict-batch to score multiple records at once. All fields are required.
          </p>
          <div className="chips">
            <span>suburb</span>
            <span>state</span>
            <span>postcode</span>
            <span>bedrooms</span>
            <span>dwelling_type</span>
            <span>date</span>
          </div>
          <p className="hint">Tip: keep the API running via <code>make serve</code>.</p>
        </section>
      </main>
    </div>
  );
}
