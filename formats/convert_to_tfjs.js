const tf = require('@tensorflow/tfjs-node');
const fs = require('fs-extra');
const path = require('path');

const OUTPUT_DIR = path.resolve(__dirname, 'output');
const SAVED_MODEL_DIR = path.join(OUTPUT_DIR, 'saved_model');
const TFJS_DIR = path.join(OUTPUT_DIR, 'tfjs');

async function convertModelsInDir(sourceDir, destBaseDir) {
    if (!fs.existsSync(sourceDir)) {
        console.log(`Directory not found, skipping: ${sourceDir}`);
        return;
    }

    const modelNames = await fs.readdir(sourceDir);

    for (const name of modelNames) {
        const modelPath = path.join(sourceDir, name);
        // Be careful to only process directories
        const stat = await fs.stat(modelPath);
        if (!stat.isDirectory()) continue;

        console.log(`Converting ${name} to TFJS...`);
        try {
            // Load the SavedModel
            const model = await tf.node.loadSavedModel(modelPath);

            // Define output path
            const outputDir = path.join(destBaseDir, name);
            await fs.ensureDir(outputDir);

            // Save to TFJS format
            await model.save(`file://${outputDir}`);
            console.log(`  ✅ Saved to ${outputDir}`);
        } catch (e) {
            console.error(`  ❌ Failed to convert ${name}:`, e.message);
        }
    }
}

async function main() {
    // 1. Convert standard models
    await convertModelsInDir(
        path.join(OUTPUT_DIR, 'saved_model'),
        path.join(OUTPUT_DIR, 'tfjs')
    );
    // 2. Convert simple models
    await convertModelsInDir(
        path.join(OUTPUT_DIR, 'simple_models/saved_model'),
        path.join(OUTPUT_DIR, 'simple_models/tfjs')
    );
}

main().then(() => console.log('Done.'));
