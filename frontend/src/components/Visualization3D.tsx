import React, { useEffect, useRef } from 'react';

interface Point3D {
  x: number;
  y: number;
  z: number;
  intensity?: number;
}

interface Visualization3DProps {
  points: Point3D[];
  width?: number;
  height?: number;
  title?: string;
}

/**
 * 3D Point Cloud Visualization Component
 * Renders 3D medical image data as an interactive point cloud
 */
const Visualization3D: React.FC<Visualization3DProps> = ({
  points,
  width = 800,
  height = 600,
  title = "3D Medical Image Reconstruction"
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rotationRef = useRef({ x: 0, y: 0 });
  const scaleRef = useRef(1);

  useEffect(() => {
    if (!canvasRef.current || points.length === 0) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Setup canvas
    canvas.width = width;
    canvas.height = height;

    // Clear canvas
    ctx.fillStyle = '#0a0e27';
    ctx.fillRect(0, 0, width, height);

    // Calculate center and bounds
    const xs = points.map(p => p.x);
    const ys = points.map(p => p.y);
    const zs = points.map(p => p.z);

    const centerX = (Math.max(...xs) + Math.min(...xs)) / 2;
    const centerY = (Math.max(...ys) + Math.min(...ys)) / 2;
    const centerZ = (Math.max(...zs) + Math.min(...zs)) / 2;

    const maxDist = Math.max(
      Math.max(...xs) - Math.min(...xs),
      Math.max(...ys) - Math.min(...ys),
      Math.max(...zs) - Math.min(...zs)
    ) / 2;

    // 3D rotation matrices
    const rotateX = (point: Point3D, angle: number) => ({
      x: point.x,
      y: point.y * Math.cos(angle) - point.z * Math.sin(angle),
      z: point.y * Math.sin(angle) + point.z * Math.cos(angle),
      intensity: point.intensity || 1
    });

    const rotateY = (point: Point3D, angle: number) => ({
      x: point.x * Math.cos(angle) + point.z * Math.sin(angle),
      y: point.y,
      z: -point.x * Math.sin(angle) + point.z * Math.cos(angle),
      intensity: point.intensity || 1
    });

    // Apply rotation and project to 2D
    const rotatedPoints = points.map(point => {
      let p = { ...point };
      p = rotateX(p, rotationRef.current.x);
      p = rotateY(p, rotationRef.current.y);
      return p;
    });

    // Project 3D to 2D (perspective projection)
    const projectedPoints = rotatedPoints.map(point => {
      const normalizedX = (point.x - centerX) / maxDist;
      const normalizedY = (point.y - centerY) / maxDist;
      const normalizedZ = (point.z - centerZ) / maxDist;

      const scale = 300 * scaleRef.current / (3 + normalizedZ);
      const x2d = width / 2 + normalizedX * scale;
      const y2d = height / 2 + normalizedY * scale;

      return {
        x: x2d,
        y: y2d,
        z: normalizedZ,
        intensity: point.intensity || 1,
        depth: normalizedZ
      };
    });

    // Sort by depth (painter's algorithm)
    projectedPoints.sort((a, b) => a.depth - b.depth);

    // Draw points
    projectedPoints.forEach(point => {
      const size = Math.max(1, 3 + point.depth * 2);
      const alpha = (point.intensity || 1) * 0.8;

      // Color gradient based on depth and intensity
      const hue = (point.depth + 1) * 60; // 0-360
      const saturation = Math.min(100, 50 + point.intensity * 50);
      const lightness = Math.min(100, 40 + point.intensity * 40);

      ctx.fillStyle = `hsla(${hue}, ${saturation}%, ${lightness}%, ${alpha})`;
      ctx.beginPath();
      ctx.arc(point.x, point.y, size, 0, Math.PI * 2);
      ctx.fill();
    });

    // Draw grid lines for reference
    ctx.strokeStyle = 'rgba(100, 100, 150, 0.2)';
    ctx.lineWidth = 1;
    for (let i = -2; i <= 2; i += 1) {
      ctx.beginPath();
      ctx.moveTo(width / 2 + i * 50, 0);
      ctx.lineTo(width / 2 + i * 50, height);
      ctx.stroke();

      ctx.beginPath();
      ctx.moveTo(0, height / 2 + i * 50);
      ctx.lineTo(width, height / 2 + i * 50);
      ctx.stroke();
    }

    // Draw axes
    const drawAxis = (p1: Point3D, p2: Point3D, color: string) => {
      const rotated1 = rotateY(rotateX(p1, rotationRef.current.x), rotationRef.current.y);
      const rotated2 = rotateY(rotateX(p2, rotationRef.current.x), rotationRef.current.y);

      const n1 = (rotated1.x - centerX) / maxDist;
      const n2 = (rotated1.y - centerY) / maxDist;
      const n3 = (rotated1.z - centerZ) / maxDist;

      const n4 = (rotated2.x - centerX) / maxDist;
      const n5 = (rotated2.y - centerY) / maxDist;
      const n6 = (rotated2.z - centerZ) / maxDist;

      const scale = 300 * scaleRef.current;
      const x1 = width / 2 + n1 * scale / (3 + n3);
      const y1 = height / 2 + n2 * scale / (3 + n3);
      const x2 = width / 2 + n4 * scale / (3 + n6);
      const y2 = height / 2 + n5 * scale / (3 + n6);

      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.stroke();
    };

    drawAxis(
      { x: centerX, y: centerY, z: centerZ },
      { x: centerX + 50, y: centerY, z: centerZ },
      '#ff4444'
    ); // X - Red
    drawAxis(
      { x: centerX, y: centerY, z: centerZ },
      { x: centerX, y: centerY + 50, z: centerZ },
      '#44ff44'
    ); // Y - Green
    drawAxis(
      { x: centerX, y: centerY, z: centerZ },
      { x: centerX, y: centerY, z: centerZ + 50 },
      '#4444ff'
    ); // Z - Blue

  }, [points, width, height]);
  // Note: rotation and scale use refs to avoid unnecessary re-renders

  // Mouse handlers for rotation
  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (e.buttons === 1) { // Left mouse button pressed
      const rect = canvasRef.current?.getBoundingClientRect();
      if (rect) {
        rotationRef.current.y += (e.movementX || 0) * 0.01;
        rotationRef.current.x += (e.movementY || 0) * 0.01;
      }
    }
  };

  const handleWheel = (e: React.WheelEvent<HTMLCanvasElement>) => {
    e.preventDefault();
    scaleRef.current *= e.deltaY > 0 ? 0.9 : 1.1;
    scaleRef.current = Math.max(0.1, Math.min(3, scaleRef.current));
  };

  const handleReset = () => {
    rotationRef.current = { x: 0, y: 0 };
    scaleRef.current = 1;
  };

  return (
    <div className="w-full bg-gradient-to-br from-slate-900 to-slate-800 rounded-lg p-6 shadow-2xl">
      <div className="mb-4">
        <h3 className="text-xl font-bold text-white mb-2">{title}</h3>
        <p className="text-sm text-gray-400">
          Drag to rotate | Scroll to zoom | Double-click to reset
        </p>
      </div>

      <canvas
        ref={canvasRef}
        className="w-full border-2 border-blue-500 rounded-lg cursor-grab active:cursor-grabbing mb-4"
        onMouseMove={handleMouseMove}
        onWheel={handleWheel}
        onDoubleClick={handleReset}
        style={{ touchAction: 'none', maxHeight: '500px' }}
      />

      <div className="flex gap-2 justify-center">
        <button
          onClick={handleReset}
          className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition"
        >
          Reset View
        </button>
        <div className="text-gray-400 text-sm flex items-center">
          Points: {points.length}
        </div>
      </div>

      <div className="mt-4 grid grid-cols-3 gap-2 text-xs text-gray-400">
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 bg-red-500 rounded"></div>
          <span>X Axis</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 bg-green-500 rounded"></div>
          <span>Y Axis</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 bg-blue-500 rounded"></div>
          <span>Z Axis</span>
        </div>
      </div>
    </div>
  );
};

export default Visualization3D;
