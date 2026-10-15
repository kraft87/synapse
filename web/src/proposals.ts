// What the Review page tells the reviewer when a decision did not go through.
import { HttpError } from './api';

// A 409 means the proposal is no longer 'proposed' (a double click, another tab, the CLI got
// there first): nothing was changed, and the pane reloads to show the status it has now. Any
// other failure says nothing about the outcome.
export const decisionProblem = (e: unknown): string =>
  e instanceof HttpError && e.status === 409
    ? 'not applied: this proposal was already decided, so the pane now shows its current status.'
    : 'decision failed: ' + String((e as { message?: string } | null)?.message ?? e);
