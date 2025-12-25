import { describe, it, expect } from "vitest";
import { render } from "@testing-library/react";
import { ImageRenderer } from "../media/ImageRenderer";
import { VideoRenderer } from "../media/VideoRenderer";
import { MarkdownRenderer } from "../media/MarkdownRenderer";

describe("Media Renderers Smoke Test", () => {
  it("renders an image", () => {
    const { getByRole } = render(<ImageRenderer url="test.jpg" />);
    expect(getByRole("img")).toBeInTheDocument();
  });

  it("renders a video", () => {
    const { container } = render(<VideoRenderer url="test.mp4" />);
    expect(container.querySelector("video")).toBeInTheDocument();
  });

  it("renders markdown", () => {
    const { getByText } = render(<MarkdownRenderer content="# Hello" />);
    expect(getByText("Hello")).toBeInTheDocument();
  });
});
