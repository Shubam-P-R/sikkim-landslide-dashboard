import React, { useState } from 'react';
import { MapContainer, TileLayer, Marker, Popup, useMapEvents } from 'react-leaflet';
import axios from 'axios';
import { 
  AlertTriangle, 
  Droplets, 
  MapPin, 
  ShieldAlert, 
  Send, 
  Loader2, 
  Info, 
  Compass, 
  Mountain 
} from 'lucide-react';
import './utils/leafletIcons';

const API_BASE_URL = 'http://localhost:8000';

// Map click listener component
function MapClickHandler({ onLocationSelect }) {
  useMapEvents({
    click(e) {
      const { lat, lng } = e.latlng;
      onLocationSelect(lat, lng);
    },
  });
  return null;
}

export default function App() {
  const [selectedCoords, setSelectedCoords] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Incident form state
  const [incidentForm, setIncidentForm] = useState({
    description: '',
    severity: 'Moderate',
  });
  const [reportSubmitting, setReportSubmitting] = useState(false);
  const [reportStatus, setReportStatus] = useState(null);

  // Triggered whenever a point on the map is clicked
  const handleLocationSelect = async (lat, lng) => {
    setSelectedCoords({ lat, lng });
    setLoading(true);
    setError(null);
    setPrediction(null);
    setReportStatus(null);

    try {
      const response = await axios.post(`${API_BASE_URL}/api/predict-risk`, {
        latitude: lat,
        longitude: lng,
      });
      setPrediction(response.data);
    } catch (err) {
      console.error('Prediction request error:', err);
      setError(err.response?.data?.detail || 'Failed to fetch risk prediction from server.');
    } finally {
      setLoading(false);
    }
  };

  // Submit incident report
  const handleReportSubmit = async (e) => {
    e.preventDefault();
    if (!selectedCoords) {
      setReportStatus({ type: 'error', message: 'Please select a coordinate on the map first.' });
      return;
    }
    if (!incidentForm.description.trim()) {
      setReportStatus({ type: 'error', message: 'Please provide a description.' });
      return;
    }

    setReportSubmitting(true);
    setReportStatus(null);

    try {
      const response = await axios.post(`${API_BASE_URL}/api/report-incident`, {
        latitude: selectedCoords.lat,
        longitude: selectedCoords.lng,
        description: incidentForm.description,
        severity: incidentForm.severity,
      });

      if (response.data.status === 'success') {
        setReportStatus({ 
          type: 'success', 
          message: `Incident recorded successfully (ID: #${response.data.id}).` 
        });
        setIncidentForm({ description: '', severity: 'Moderate' });
      }
    } catch (err) {
      console.error('Incident report error:', err);
      setReportStatus({
        type: 'error',
        message: err.response?.data?.detail || 'Failed to submit incident report.',
      });
    } finally {
      setReportSubmitting(false);
    }
  };

  const getTierColor = (tier) => {
    switch (tier?.toLowerCase()) {
      case 'low':
        return {
          badge: 'bg-emerald-100 text-emerald-800 border-emerald-300',
          bar: 'bg-emerald-500',
          text: 'text-emerald-700',
        };
      case 'moderate':
        return {
          badge: 'bg-amber-100 text-amber-800 border-amber-300',
          bar: 'bg-amber-500',
          text: 'text-amber-700',
        };
      case 'high':
        return {
          badge: 'bg-orange-100 text-orange-800 border-orange-300',
          bar: 'bg-orange-500',
          text: 'text-orange-700',
        };
      case 'critical':
      default:
        return {
          badge: 'bg-rose-100 text-rose-800 border-rose-300',
          bar: 'bg-rose-500',
          text: 'text-rose-700',
        };
    }
  };

  const tierColors = prediction ? getTierColor(prediction.tier) : null;

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-slate-100">
      {/* Sidebar Panel */}
      <aside className="w-[420px] flex-shrink-0 h-full overflow-y-auto bg-white border-r border-slate-200 p-6 flex flex-col justify-between shadow-lg z-10">
        <div className="space-y-6">
          {/* Header */}
          <div className="border-b border-slate-200 pb-4">
            <div className="flex items-center space-x-2 text-indigo-700">
              <ShieldAlert className="w-7 h-7" />
              <h1 className="text-xl font-bold tracking-tight text-slate-900">
                Landslide Early Warning
              </h1>
            </div>
            <p className="text-xs text-slate-500 mt-1">
              Sikkim Geospatial ML Susceptibility & Risk Engine
            </p>
          </div>

          {/* Selected Coordinates */}
          <div className="bg-slate-50 border border-slate-200 rounded-xl p-3.5 text-sm">
            <div className="flex items-center justify-between text-slate-600 mb-1">
              <span className="flex items-center text-xs font-semibold uppercase tracking-wider text-slate-500">
                <MapPin className="w-3.5 h-3.5 mr-1" /> Clicked Location
              </span>
              {selectedCoords && (
                <span className="text-[11px] bg-indigo-50 text-indigo-700 px-2 py-0.5 rounded font-mono font-medium">
                  Active Spot
                </span>
              )}
            </div>
            {selectedCoords ? (
              <div className="font-mono text-slate-800 text-xs flex justify-between pt-1">
                <span>Lat: {selectedCoords.lat.toFixed(5)}</span>
                <span>Lng: {selectedCoords.lng.toFixed(5)}</span>
              </div>
            ) : (
              <p className="text-xs text-slate-400 italic pt-1">
                Click anywhere on the map to evaluate risk...
              </p>
            )}
          </div>

          {/* Inference Status / Error */}
          {loading && (
            <div className="flex items-center justify-center space-x-3 bg-indigo-50 border border-indigo-200 rounded-xl p-4 text-indigo-700">
              <Loader2 className="w-5 h-5 animate-spin" />
              <span className="text-sm font-medium">Querying spatial features & running model...</span>
            </div>
          )}

          {error && (
            <div className="flex items-start space-x-2 bg-rose-50 border border-rose-200 rounded-xl p-3 text-rose-700 text-xs">
              <AlertTriangle className="w-4 h-4 mt-0.5 flex-shrink-0" />
              <span>{error}</span>
            </div>
          )}

          {/* Risk Score & Evaluation Card */}
          {prediction && !loading && (
            <div className="bg-white border border-slate-200 rounded-2xl p-5 shadow-sm space-y-4">
              <div className="flex justify-between items-start">
                <div>
                  <span className="text-xs font-semibold uppercase tracking-wider text-slate-500">
                    Susceptibility Score
                  </span>
                  <div className="flex items-baseline space-x-2 mt-0.5">
                    <span className="text-3xl font-extrabold text-slate-900">
                      {(prediction.risk_score * 100).toFixed(1)}%
                    </span>
                    <span className="text-xs text-slate-400 font-mono">
                      ({prediction.risk_score.toFixed(4)})
                    </span>
                  </div>
                </div>

                <span
                  className={`px-3 py-1 rounded-full text-xs font-bold border uppercase tracking-wider ${tierColors.badge}`}
                >
                  {prediction.tier}
                </span>
              </div>

              {/* Progress bar */}
              <div className="w-full bg-slate-100 rounded-full h-2.5 overflow-hidden">
                <div
                  className={`h-full transition-all duration-500 ${tierColors.bar}`}
                  style={{ width: `${Math.min(prediction.risk_score * 100, 100)}%` }}
                />
              </div>

              {/* Rainfall & Environmental Metrics */}
              <div className="grid grid-cols-2 gap-2.5 pt-1 text-xs">
                <div className="bg-sky-50 border border-sky-100 rounded-xl p-2.5 flex items-center space-x-2">
                  <Droplets className="w-5 h-5 text-sky-600 flex-shrink-0" />
                  <div>
                    <div className="text-[11px] text-slate-500 font-medium">24h Rainfall</div>
                    <div className="font-bold text-slate-800">
                      {prediction.live_rainfall_mm !== null
                        ? `${prediction.live_rainfall_mm} mm`
                        : `${prediction.features?.RAINFALL1 || 'N/A'} mm (DB)`}
                    </div>
                  </div>
                </div>

                <div className="bg-emerald-50 border border-emerald-100 rounded-xl p-2.5 flex items-center space-x-2">
                  <Mountain className="w-5 h-5 text-emerald-600 flex-shrink-0" />
                  <div>
                    <div className="text-[11px] text-slate-500 font-medium">Elevation</div>
                    <div className="font-bold text-slate-800">
                      {prediction.features?.ELEVATION1
                        ? `${Math.round(prediction.features.ELEVATION1)} m`
                        : 'N/A'}
                    </div>
                  </div>
                </div>
              </div>

              {/* Nearest Data Details */}
              <div className="bg-slate-50 rounded-lg p-2.5 text-[11px] text-slate-600 space-y-1">
                <div className="flex justify-between">
                  <span className="text-slate-400">Nearest Station:</span>
                  <span className="font-mono font-medium">
                    {prediction.nearest_coordinate?.latitude.toFixed(4)},{' '}
                    {prediction.nearest_coordinate?.longitude.toFixed(4)}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Slope:</span>
                  <span className="font-medium">{prediction.features?.SLOPE1 ?? 'N/A'}°</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">NDVI:</span>
                  <span className="font-medium">{prediction.features?.NDVI1 ?? 'N/A'}</span>
                </div>
              </div>
            </div>
          )}

          {/* Report Incident Form */}
          <div className="bg-white border border-slate-200 rounded-2xl p-5 shadow-sm space-y-3.5">
            <div className="flex items-center space-x-1.5 text-slate-900 font-semibold text-sm">
              <AlertTriangle className="w-4 h-4 text-amber-500" />
              <span>Report Local Incident</span>
            </div>
            <p className="text-xs text-slate-500">
              Submit observations or slope stability concerns at the selected coordinates.
            </p>

            <form onSubmit={handleReportSubmit} className="space-y-3">
              <div>
                <label className="block text-xs font-medium text-slate-700 mb-1">
                  Severity Level
                </label>
                <select
                  value={incidentForm.severity}
                  onChange={(e) => setIncidentForm({ ...incidentForm, severity: e.target.value })}
                  className="w-full text-xs border border-slate-300 rounded-lg px-2.5 py-2 focus:ring-2 focus:ring-indigo-500 focus:outline-none bg-white text-slate-800"
                >
                  <option value="Low">Low (Surface erosion, minor rubble)</option>
                  <option value="Moderate">Moderate (Cracks on slope / roadside)</option>
                  <option value="High">High (Active mudflow, partial blockage)</option>
                  <option value="Critical">Critical (Total mass movement / road blocked)</option>
                </select>
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-700 mb-1">
                  Description
                </label>
                <textarea
                  rows="2"
                  value={incidentForm.description}
                  onChange={(e) => setIncidentForm({ ...incidentForm, description: e.target.value })}
                  placeholder="e.g., Fissures observed on slope after rainfall..."
                  className="w-full text-xs border border-slate-300 rounded-lg p-2.5 focus:ring-2 focus:ring-indigo-500 focus:outline-none text-slate-800"
                />
              </div>

              {reportStatus && (
                <div
                  className={`text-xs p-2.5 rounded-lg border ${
                    reportStatus.type === 'success'
                      ? 'bg-emerald-50 text-emerald-800 border-emerald-200'
                      : 'bg-rose-50 text-rose-800 border-rose-200'
                  }`}
                >
                  {reportStatus.message}
                </div>
              )}

              <button
                type="submit"
                disabled={reportSubmitting || !selectedCoords}
                className="w-full flex items-center justify-center space-x-2 bg-indigo-600 hover:bg-indigo-700 text-white text-xs font-semibold py-2.5 px-4 rounded-lg transition disabled:opacity-50 disabled:cursor-not-allowed shadow-sm"
              >
                {reportSubmitting ? (
                  <>
                    <Loader2 className="w-3.5 h-3.5 animate-spin" />
                    <span>Submitting Report...</span>
                  </>
                ) : (
                  <>
                    <Send className="w-3.5 h-3.5" />
                    <span>Submit Incident Report</span>
                  </>
                )}
              </button>
            </form>
          </div>
        </div>

        {/* Footer */}
        <div className="pt-4 border-t border-slate-200 text-center text-[11px] text-slate-400">
          FastAPI &bull; Neon PostgreSQL &bull; Open-Meteo &bull; Leaflet
        </div>
      </aside>

      {/* Main Map Container */}
      <main className="flex-1 h-full relative">
        <MapContainer
          center={[27.5330, 88.5122]}
          zoom={10}
          scrollWheelZoom={true}
          className="h-full w-full"
        >
          <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          />

          <MapClickHandler onLocationSelect={handleLocationSelect} />

          {selectedCoords && (
            <Marker position={[selectedCoords.lat, selectedCoords.lng]}>
              <Popup>
                <div className="p-1 space-y-1 text-xs">
                  <div className="font-bold text-slate-900">Selected Spot</div>
                  <div>Lat: {selectedCoords.lat.toFixed(4)}</div>
                  <div>Lng: {selectedCoords.lng.toFixed(4)}</div>
                  {prediction && (
                    <div className="mt-1 pt-1 border-t border-slate-200">
                      <span className="font-semibold">Risk: </span>
                      <span className={tierColors?.text}>
                        {prediction.tier} ({(prediction.risk_score * 100).toFixed(1)}%)
                      </span>
                    </div>
                  )}
                </div>
              </Popup>
            </Marker>
          )}
        </MapContainer>
      </main>
    </div>
  );
}
