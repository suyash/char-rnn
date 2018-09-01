export function bernoulli(p: number): number {
    return (Math.random() < p) ? 1 : 0;
}

export function discrete(p: Float32Array): number {
    for (let i: number = 0 ; i < p.length ; i++) {
        const x: number = p[i] / p.slice(i).reduce((a: number, b: number): number => a + b, 0);
        if (bernoulli(x)) {
            return i;
        }
    }

    return p.length - 1;
}
