export function rushMultiplier() {
  const hour = new Date().getHours();

  // Morning + evening rush
  if (hour >= 7 && hour <= 9) return 2.2;
  if (hour >= 17 && hour <= 19) return 2.5;

  return 1;
}

export function heatColor(v) {
  if (v < 8) return "#22c55e";
  if (v < 16) return "#facc15";
  return "#ef4444";
}
