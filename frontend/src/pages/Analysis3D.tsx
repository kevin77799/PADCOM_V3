import React, { useState } from 'react';
import { useDropzone } from 'react-dropzone';
import Visualization3D from '../components/Visualization3D';

interface AnalysisResult {
  analysis_type: string;
  analysis_2d: {
    prediction: string;
    confidence: number;
    analysis_text: string;
    model: string;
  };
  reconstruction_3d: {
    coordinates: any;
    point_cloud_size: number;
    features: any;
  };
  visualization_data: {
    mesh_points: Array<{ x: number; y: number; z: number; intensity: number }>;
    point_cloud: number[][];
    intensities: number[];
  };
}

const Analysis3D: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [analysisType, setAnalysisType] = useState<'brain_tumor' | 'pneumonia'>('brain_tumor');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const onDrop = (acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      const file = acceptedFiles[0];
      if (file.type.startsWith('image/')) {
        setSelectedFile(file);
        setError(null);
      } else {
        setError('Please upload a valid image file');
      }
    }
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'image/*': ['.jpeg', '.jpg', '.png'] }
  });

  const handleAnalyze = async () => {
    if (!selectedFile) {
      setError('Please select an image first');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('image', selectedFile);
      formData.append('analysis_type', analysisType);

      const response = await fetch('http://localhost:5000/api/3d/analyze-and-reconstruct', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        throw new Error(`Analysis failed: ${response.statusText}`);
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Analysis failed');
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  const convertPointCloudTo3D = (pointCloud: number[][], intensities: number[]) => {
    return pointCloud.map((point, idx) => ({
      x: point[0],
      y: point[1],
      z: point[2],
      intensity: intensities[idx] || 0.5
    }));
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin text-4xl mb-4">🔄</div>
          <p className="text-white text-lg">Analyzing image and generating 3D reconstruction...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 p-6">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-white mb-2">3D Medical Image Analysis</h1>
          <p className="text-gray-400">
            Advanced 2D-to-3D conversion with Ollama AI-powered analysis
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Control Panel */}
          <div className="lg:col-span-1 space-y-4">
            <div className="bg-slate-800 rounded-lg p-6 border border-slate-700">
              <h2 className="text-xl font-bold text-white mb-4">Analysis Settings</h2>

              {/* Analysis Type Selection */}
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Analysis Type
                </label>
                <select
                  value={analysisType}
                  onChange={(e) => setAnalysisType(e.target.value as 'brain_tumor' | 'pneumonia')}
                  className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded text-white"
                >
                  <option value="brain_tumor">Brain Tumor (MRI)</option>
                  <option value="pneumonia">Pneumonia (X-ray)</option>
                </select>
              </div>

              {/* File Upload */}
              <div
                {...getRootProps()}
                className={`border-2 border-dashed rounded-lg p-6 text-center cursor-pointer transition ${
                  isDragActive
                    ? 'border-blue-500 bg-blue-500 bg-opacity-10'
                    : 'border-gray-600 hover:border-gray-500'
                }`}
              >
                <input {...getInputProps()} />
                <div className="text-gray-400">
                  {selectedFile ? (
                    <>
                      <p className="font-medium text-green-400">✓ {selectedFile.name}</p>
                      <p className="text-sm mt-1">
                        {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                      </p>
                    </>
                  ) : (
                    <>
                      <p className="font-medium">Drop image here or click to select</p>
                      <p className="text-sm mt-1">PNG, JPG, JPEG</p>
                    </>
                  )}
                </div>
              </div>

              {/* Analyze Button */}
              <button
                onClick={handleAnalyze}
                disabled={!selectedFile || loading}
                className="w-full mt-4 px-4 py-3 bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800 disabled:opacity-50 disabled:cursor-not-allowed text-white font-bold rounded-lg transition"
              >
                {loading ? 'Analyzing...' : 'Analyze & Reconstruct'}
              </button>

              {/* Error Display */}
              {error && (
                <div className="mt-4 p-4 bg-red-500 bg-opacity-20 border border-red-500 rounded text-red-300 text-sm">
                  {error}
                </div>
              )}
            </div>

            {/* Results Info */}
            {result && (
              <div className="bg-slate-800 rounded-lg p-6 border border-slate-700">
                <h3 className="text-lg font-bold text-white mb-4">Analysis Results</h3>

                <div className="space-y-3 text-sm">
                  <div>
                    <p className="text-gray-400">Prediction</p>
                    <p className="text-white font-bold">
                      {result.analysis_2d?.prediction}
                    </p>
                  </div>

                  <div>
                    <p className="text-gray-400">Confidence</p>
                    <div className="flex items-center gap-2">
                      <div className="flex-1 bg-slate-700 rounded h-2 overflow-hidden">
                        <div
                          className="bg-gradient-to-r from-green-500 to-blue-500 h-full"
                          style={{ width: `${(result.analysis_2d?.confidence || 0) * 100}%` }}
                        />
                      </div>
                      <span className="text-white font-bold">
                        {Math.round((result.analysis_2d?.confidence || 0) * 100)}%
                      </span>
                    </div>
                  </div>

                  <div>
                    <p className="text-gray-400">AI Model</p>
                    <p className="text-white text-xs break-all">
                      {result.analysis_2d?.model || 'unknown'}
                    </p>
                  </div>

                  {result.reconstruction_3d && (
                    <>
                      <div>
                        <p className="text-gray-400">3D Points</p>
                        <p className="text-white font-bold">
                          {result.reconstruction_3d.point_cloud_size?.toLocaleString() || 0}
                        </p>
                      </div>

                      {result.reconstruction_3d.coordinates && (
                        <div>
                          <p className="text-gray-400">Reconstruction Quality</p>
                          <p className="text-white font-bold">
                            {(result.reconstruction_3d.coordinates.reconstruction_quality * 100).toFixed(1)}%
                          </p>
                        </div>
                      )}
                    </>
                  )}
                </div>
              </div>
            )}
          </div>

          {/* Visualization Area */}
          <div className="lg:col-span-2">
            {result && result.visualization_data ? (
              <>
                <Visualization3D
                  points={convertPointCloudTo3D(
                    result.visualization_data.point_cloud || [],
                    result.visualization_data.intensities || []
                  )}
                  width={700}
                  height={600}
                  title={`3D ${analysisType === 'brain_tumor' ? 'Brain Tumor' : 'Pneumonia'} Reconstruction`}
                />

                {/* Analysis Text */}
                {result.analysis_2d?.analysis_text && (
                  <div className="mt-6 bg-slate-800 rounded-lg p-6 border border-slate-700">
                    <h3 className="text-lg font-bold text-white mb-3">Detailed Analysis</h3>
                    <p className="text-gray-300 leading-relaxed text-sm">
                      {result.analysis_2d.analysis_text}
                    </p>
                  </div>
                )}
              </>
            ) : (
              <div className="bg-slate-800 rounded-lg p-12 border-2 border-dashed border-slate-700 flex items-center justify-center h-full">
                <div className="text-center">
                  <div className="text-5xl mb-4">📊</div>
                  <p className="text-gray-400">
                    Upload an image and click "Analyze & Reconstruct" to see the 3D visualization
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Analysis3D;
