/**
 * Bounds constraint on the variables.
 *
 * The constraint has the general inequality form:
 *   lb <= x <= ub
 */
export class Bounds {
  public lb: number[];
  public ub: number[];
  public keepFeasible: boolean[];

  constructor(
    lb: number | number[] = -Infinity,
    ub: number | number[] = Infinity,
    keepFeasible: boolean | boolean[] = true,
  ) {
    this.lb = Array.isArray(lb) ? lb : [lb];
    this.ub = Array.isArray(ub) ? ub : [ub];
    this.keepFeasible = Array.isArray(keepFeasible)
      ? keepFeasible
      : Array(this.lb.length).fill(keepFeasible);

    if (this.lb.length !== this.ub.length) {
      throw new Error("`lb` and `ub` must have the same length.");
    }
    if (this.keepFeasible.length < this.lb.length) {
      const kf = this.keepFeasible[0] ?? true;
      this.keepFeasible = Array(this.lb.length).fill(kf);
    }
  }

  get length(): number {
    return this.lb.length;
  }

  /**
   * Clip value to stay within bounds at the given index.
   */
  clip(index: number, value: number): number {
    return Math.max(this.lb[index], Math.min(this.ub[index], value));
  }

  /**
   * Clip all values in an array to stay within bounds.
   */
  clipAll(values: number[]): number[] {
    return values.map((v, i) => this.clip(i, v));
  }

  /**
   * Calculate residuals (slack) between input and bounds.
   */
  residual(x: number[]): { sl: number[]; sb: number[] } {
    const sl = x.map((v, i) => v - this.lb[i]);
    const sb = x.map((v, i) => this.ub[i] - v);
    return { sl, sb };
  }

  toJSON(): { lb: number[]; ub: number[]; keepFeasible: boolean[] } {
    return {
      lb: this.lb,
      ub: this.ub,
      keepFeasible: this.keepFeasible,
    };
  }
}
