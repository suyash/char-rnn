import { discrete } from "./sampling";

function codeToVocab(code: number): number {
    if (code === 9) {
        return 1;
    } else if (code === 10) {
        return 127 - 30;
    } else if (32 <= code && code <= 126) {
        return code - 30;
    }

    return 0;
}

export function encode(s: string): number[] {
    const ans: number[] = Array.from({ length: s.length });

    for (let i: number = 0 ; i < s.length ; i++) {
        ans[i] = codeToVocab(s.charCodeAt(i));
    }

    return ans;
}

export function decode(code: number): string {
    if (code === 1) {
        return String.fromCharCode(9);
    } else if (code === 127 - 30) {
        return String.fromCharCode(10);
    } else if (32 <= (code + 30) && (code + 30) <= 126) {
        return String.fromCharCode(code + 30);
    }

    return String.fromCharCode(0);
}

export function sample(p: Float32Array): number {
    return discrete(p);
}

export function sleep(ms: number): Promise<void> {
    return new Promise((resolve: () => void): void => {
        setTimeout(resolve, ms);
    });
}
