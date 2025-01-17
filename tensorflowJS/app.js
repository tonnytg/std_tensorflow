// Import TensorFlow.js and necessary Node.js modules
import * as tf from '@tensorflow/tfjs-node';
import * as fs from 'fs';
import * as path from 'path';

// Define the file path to the image
const FILE_PATH = 'files';
const cakeImagePath = path.join(FILE_PATH, 'cake.png');

// Read the image file into a buffer
const cakeImage = fs.readFileSync(cakeImagePath);

tf.tidy(async () => {
  // Decode the image into a tensor (RGBA format by default)
  const cakeTensor = tf.node.decodeImage(cakeImage);
  console.log(`Success: Local file converted to a tensor with shape ${cakeTensor.shape}`);

  // Analyze the image dimensions
  const [height, width, channels] = cakeTensor.shape;
  console.log(`The cake image is ${width} pixels wide and ${height} pixels tall.`);

  // Calculate the average color value (only RGB channels)
  const rgbTensor = cakeTensor.slice([0, 0, 0], [-1, -1, 3]); // Ignore alpha channel
  const meanColor = rgbTensor.mean([0, 1]); // Mean along height and width
  const meanValues = meanColor.arraySync();
  console.log(`Average RGB color values: R=${meanValues[0].toFixed(2)}, G=${meanValues[1].toFixed(2)}, B=${meanValues[2].toFixed(2)}`);

  // Display a message based on average brightness
  const brightness = (meanValues[0] + meanValues[1] + meanValues[2]) / 3;
  if (brightness > 150) {
    console.log("This cake looks bright and delightful! 🍰");
  } else if (brightness > 100) {
    console.log("This cake looks perfectly balanced. Delicious! 😋");
  } else {
    console.log("This cake looks dark and mysterious. Is it chocolate? 🍫");
  }

  // Convert the image to grayscale
  const cakeBWTensor = tf.node.decodeImage(cakeImage, 1);
  console.log(`Grayscale version created with shape ${cakeBWTensor.shape}`);

  // Save the grayscale image as a new file
  const grayscaleImagePath = path.join(FILE_PATH, 'cake_grayscale.png');
  const grayscaleBuffer = await tf.node.encodeJpeg(cakeBWTensor); // Await the promise
  fs.writeFileSync(grayscaleImagePath, grayscaleBuffer);
  console.log(`Grayscale image saved to: ${grayscaleImagePath}`);
});

