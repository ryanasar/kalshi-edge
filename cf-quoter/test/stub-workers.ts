export class DurableObject<E = unknown> {
  constructor(public ctx: any, public env: E) {}
}
