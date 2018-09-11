import * as tf from "@tensorflow/tfjs";
import debug from "debug";

import { decode, encode, sample, sleep } from "./utils";

debug.enable("*");

const INITIAL_TEXT: string = "Immortal longings in me: now no more";

let speed: number = 38;
let playing: boolean = false;
let text: string = INITIAL_TEXT;

let predictionContainer: HTMLElement;
let speedElement: HTMLInputElement;
let textElement: HTMLPreElement;
let button: HTMLButtonElement;

window.addEventListener("DOMContentLoaded", main);

async function main(): Promise<void> {
    const log: debug.IDebugger = debug("main");

    predictionContainer = document.querySelector("#predictionContainer") as HTMLElement;
    speedElement = document.querySelector("#speedControlSlider") as HTMLInputElement;
    textElement = document.querySelector("pre") as HTMLInputElement;
    button = document.querySelector("#controls button") as HTMLButtonElement;

    (speedElement as any).value = speed;
    speedElement.addEventListener("change", onSpeedChange);

    button.addEventListener("click", () => onButtonClick(model));

    textElement.innerText = text;

    let model: tf.Model;

    try {
        model = await tf.loadModel("indexeddb://model_shakespeare");
        log("loaded from idb");
    } catch (err) {
        model = await tf.loadModel("model/shakespeare/v1/model.json");
        const result: tf.io.SaveResult = await model.save("indexeddb://model_shakespeare");
        log("saving to idb:", result);
    }

    model.summary();

    (document.querySelector("#loadingMessage") as HTMLElement).classList.add("hidden");
    (document.querySelector("#controls") as HTMLElement).classList.remove("hidden");
    predictionContainer.classList.remove("hidden");

    setTimeout(() => textElement.focus(), 0);
}

function onSpeedChange(e: Event): void {
    const log: debug.IDebugger = debug("onSpeedChange");
    speed = (e.target as any).value;
    log("speed changed to", speed);
}

function onButtonClick(model: tf.Model): void {
    if (playing) {
        playing = false;
        button.innerText = "Start";
    } else {
        playing = true;

        button.classList.add("hidden");
        predictionContainer.classList.remove("editable");
        textElement.removeAttribute("contenteditable");

        predict(model);
    }
}

async function predict(model: tf.Model): Promise<void> {
    while (playing) {
        const [nc]: [string, void] = await Promise.all([
            next(text, model),
            sleep((41 - speed) * 6),
        ]);

        textElement.innerText += nc;
        text = text.substr(1) + nc;
    }
}

async function next(c: string, model: tf.Model): Promise<string> {
    const predictions: tf.Tensor<tf.Rank.R1> = tf.tidy(() => generatePredictions(c, model));
    const data: Float32Array = await predictions.data() as Float32Array;
    predictions.dispose();

    const nextCode: number = sample(data);
    return decode(nextCode);
}

function generatePredictions(c: string, model: tf.Model): tf.Tensor<tf.Rank.R1> {
    const batchSize: number = (model.input as tf.SymbolicTensor).shape[0];
    const sequenceLength: number = c.length;
    const vocabSize: number = (model.input as tf.SymbolicTensor).shape[2];

    const inp: tf.Tensor<tf.Rank.R3> = tf.tidy(() => generateInput(c, batchSize, vocabSize));

    const predictions: tf.Tensor<tf.Rank> = model.predict(inp, { batchSize }) as tf.Tensor<tf.Rank>;

    const lastPrediction: tf.Tensor<tf.Rank.R3> = tf.slice3d(
        predictions as tf.Tensor<tf.Rank.R3>,
        [0, sequenceLength - 1, 0],
        [batchSize, 1, vocabSize],
    );

    return lastPrediction.reshape([batchSize, vocabSize]).sum(0);
}

function generateInput(c: string, batchSize: number, vocabSize: number): tf.Tensor<tf.Rank.R3> {
    const code: number[] = encode(c);
    const tensor: tf.Tensor<tf.Rank.R2> = tf.oneHot(code, vocabSize);
    return tf.stack(Array.from({ length: batchSize }).map(() => tensor)) as tf.Tensor<tf.Rank.R3>;
}
