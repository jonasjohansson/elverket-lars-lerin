export const SURFACE_NAMES = [
  "Front_LongWallA",
  "Rear_LongWallB",
  "Left_ShortWallC",
  "Right_ShortWallD",
  "Floor",
];

export const ROOM = {
  floor: 4.62,
  ceiling: 12.60,
  xMin: 3.13,
  xMax: 14.45,
  zMin: -17.29,
  zMax: 16.79,
  get centerX() { return (this.xMin + this.xMax) / 2; },
  get centerY() { return (this.floor + this.ceiling) / 2; },
  get centerZ() { return (this.zMin + this.zMax) / 2; },
  get width()   { return this.xMax - this.xMin; },
  get depth()   { return this.zMax - this.zMin; },
  get height()  { return this.ceiling - this.floor; },
};

export const CAMERA_HEIGHT = 1.7;
