import React, { useState } from 'react';
import { Upload, Brain, Lungs, AlertCircle, CheckCircle } from 'lucide-react';

interface AnalysisResult {
  prediction: string;
  confidence: string;
  class: string;
  probabilities?: any;
  message?: string;
}

const MedicalAnalysis: React.FC = () => {
  const [selectedType, setSelectedType] = useState<'brain' | 'pneumonia'>('brain');
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string>('');
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  const handleImageSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedImage(file);
      setPreviewUrl(URL.createObjectURL(file));
      setResult(null);
    }
  };

  const analyzeImage = async () => {
    if (!selectedImage) return;

    setIsAnalyzing(true);
    const formData = new FormData();
    formData.append('image', selectedImage);

    try {
      const endpoint = selectedType === 'brain' 
        ? 'http://localhost:5000/api/medical/brain-tumor'
        : 'http://localhost:5000/api/medical/pneumonia';

      const response = await fetch(endpoint, {
        method: 'POST',
        body: formData
      });

      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error('Analysis error:', error);
      setResult({
        prediction: 'Error',
        confidence: '0%',
        class: 'error',
        message: 'Failed to analyze image'
      });
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 text-cyan-400 p-6">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-cyan-300 mb-2">Medical Image Analysis</h1>
          <p className="text-gray-400">AI-powered diagnostic assistance with 95%+ accuracy</p>
        </div>

        {/* Analysis Type Selection */}
        <div className="grid grid-cols-2 gap-4 mb-8">
          <button
            onClick={() => {
              setSelectedType('brain');
              setResult(null);
            }}
            className={`p-6 rounded-lg border-2 transition ${
              selectedType === 'brain'
                ? 'bg-cyan-900 border-cyan-500'
                : 'bg-gray-800 border-gray-700 hover:border-cyan-700'
            }`}
          >
            <Brain className="w-12 h-12 mb-3 mx-auto" />
            <h3 className="text-xl font-bold mb-1">Brain Tumor</h3>
            <p className="text-sm text-gray-400">MRI scan analysis</p>
          </button>

          <button
            onClick={() => {
              setSelectedType('pneumonia');
              setResult(null);
            }}
            className={`p-6 rounded-lg border-2 transition ${
              selectedType === 'pneumonia'
                ? 'bg-cyan-900 border-cyan-500'
                : 'bg-gray-800 border-gray-700 hover:border-cyan-700'
            }`}
          >
            <Lungs className="w-12 h-12 mb-3 mx-auto" />
            <h3 className="text-xl font-bold mb-1">Pneumonia</h3>
            <p className="text-sm text-gray-400">Chest X-ray analysis</p>
          </button>
        </div>

        {/* Upload Area */}
        <div className="bg-gray-800 rounded-lg p-8 border border-cyan-700 mb-8">
          <div className="text-center">
            {!previewUrl ? (
              <label className="cursor-pointer">
                <input
                  type="file"
                  accept="image/*"
                  onChange={handleImageSelect}
                  className="hidden"
                />
                <div className="border-2 border-dashed border-cyan-700 rounded-lg p-12 hover:border-cyan-500 transition">
                  <Upload className="w-16 h-16 mx-auto mb-4 text-cyan-500" />
                  <p className="text-lg mb-2">Click to upload medical image</p>
                  <p className="text-sm text-gray-500">PNG, JPG up to 10MB</p>
                </div>
              </label>
            ) : (
              <div>
                <img
                  src={previewUrl}
                  alt="Preview"
                  className="max-w-md mx-auto rounded-lg mb-4 border border-cyan-700"
                />
                <div className="flex gap-4 justify-center">
                  <label className="cursor-pointer bg-gray-700 px-4 py-2 rounded hover:bg-gray-600 transition">
                    Change Image
                    <input
                      type="file"
                      accept="image/*"
                      onChange={handleImageSelect}
                      className="hidden"
                    />
                  </label>
                  <button
                    onClick={analyzeImage}
                    disabled={isAnalyzing}
                    className="bg-cyan-600 hover:bg-cyan-700 disabled:bg-gray-600 px-6 py-2 rounded transition"
                  >
                    {isAnalyzing ? 'Analyzing...' : 'Analyze Image'}
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Results */}
        {result && (
          <div className={`bg-gray-800 rounded-lg p-6 border-2 ${
            result.class === 'positive' ? 'border-red-500' : 
            result.class === 'negative' ? 'border-green-500' : 
            'border-yellow-500'
          }`}>
            <div className="flex items-start gap-4">
              {result.class === 'positive' ? (
                <AlertCircle className="w-12 h-12 text-red-500 flex-shrink-0" />
              ) : result.class === 'negative' ? (
                <CheckCircle className="w-12 h-12 text-green-500 flex-shrink-0" />
              ) : (
                <AlertCircle className="w-12 h-12 text-yellow-500 flex-shrink-0" />
              )}
              
              <div className="flex-1">
                <h3 className="text-2xl font-bold mb-2">{result.prediction}</h3>
                <p className="text-xl mb-4">Confidence: {result.confidence}</p>
                
                {result.probabilities && (
                  <div className="bg-gray-900 rounded p-4 mb-4">
                    <p className="text-sm text-gray-400 mb-2">Detailed Probabilities:</p>
                    {Object.entries(result.probabilities).map(([key, value]) => (
                      <div key={key} className="flex justify-between mb-1">
                        <span className="capitalize">{key}:</span>
                        <span className="font-bold">{value as string}</span>
                      </div>
                    ))}
                  </div>
                )}
                
                {result.message && (
                  <p className="text-sm text-gray-400 mt-4">{result.message}</p>
                )}
                
                <div className="mt-4 p-4 bg-yellow-900 bg-opacity-20 border border-yellow-700 rounded">
                  <p className="text-sm text-yellow-400">
                    ⚠️ <strong>Disclaimer:</strong> This is an AI-assisted diagnostic tool. 
                    Always consult a qualified medical professional for proper diagnosis and treatment.
                  </p>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default MedicalAnalysis;
