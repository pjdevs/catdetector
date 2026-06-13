import { render, screen } from "@testing-library/svelte";
import { describe, expect, it } from "vitest";

import App from "./App.svelte";

describe("App", () => {
  it("mounts the photo analysis UI", () => {
    render(App);

    expect(
      screen.getByRole("heading", { name: "Vickie or Oka?" }),
    ).toBeTruthy();
    expect(screen.getByText("Take or choose a photo")).toBeTruthy();
    expect(screen.getByRole("button", { name: "Analyze" })).toBeTruthy();
  });
});
