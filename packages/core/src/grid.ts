import { CMASleepWakeModel } from './cma.js';

export class PFunCMAParamsGrid {
    N: number;
    m: number;
    keys: string[];
    include_mealtimes: boolean;
    collection: any[] = [];
    pgrid: any[] = [];

    private readonly PARAM_RANGES: Record<string, [number, number]> = {
        taug: [0.1, 3.0],
        taup: [0.5, 3.0],
        B: [0.0, 1.0],
        Cm: [0.0, 2.0],
        d: [-12.0, 14.0],
        toff: [-3.0, 3.0]
    };

    constructor(options: { N?: number, m?: number, keys?: string[], include_mealtimes?: boolean } = {}) {
        this.N = options.N ?? 6;
        this.m = options.m ?? 3;
        this.keys = options.keys ?? ["taug", "taup", "B", "Cm"];
        this.include_mealtimes = options.include_mealtimes ?? true;
        
        // Dynamically build the parameter grid based on m span and requested keys
        this.buildGrid();
    }

    private buildGrid() {
        const grid: Record<string, number[]> = {};
        for (const key of this.keys) {
            if (this.PARAM_RANGES[key]) {
                const [min, max] = this.PARAM_RANGES[key];
                const step = (max - min) / (this.m - 1);
                grid[key] = Array.from({ length: this.m }, (_, i) => min + step * i);
            }
        }

        // Generate cartesian product of the ranges
        const cartesian = (arrays: number[][]): number[][] => {
            return arrays.reduce((acc, curr) => 
                acc.flatMap(c => curr.map(n => [...c, n])), 
                [[]] as number[][]
            );
        };

        const keyNames = Object.keys(grid);
        const ranges = Object.values(grid);
        
        if (keyNames.length > 0) {
            const combinations = cartesian(ranges);
            this.pgrid = combinations.map(combination => {
                const obj: Record<string, number> = {};
                combination.forEach((val, i) => {
                    obj[keyNames[i]] = val;
                });
                return obj;
            });
        }
    }

    run(): void {
        this.collection = [];
        for (const params of this.pgrid) {
            const model = new CMASleepWakeModel({ ...params, N: this.N });
            model.solve();
            this.collection.push({ ...model.params, ...model.solution });
        }
    }
}
