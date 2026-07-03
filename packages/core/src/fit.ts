import { Database } from "duckdb";
import PLS from "ml-pls";

export class ModelFitter {
	private db: Database;

	constructor(dbPath: string = ":memory:") {
		this.db = new Database(dbPath);
	}

	public async loadData(csvPath: string) {
		return new Promise((resolve, reject) => {
			this.db.all(
				`SELECT * FROM read_csv_auto('${csvPath}')`,
				(err, res) => {
					if (err) reject(err);
					else resolve(res);
				},
			);
		});
	}

	public fitModel(X: number[][], Y: number[][]) {
		const pls = new PLS();
		pls.train(X, Y);
		return pls;
	}
}
