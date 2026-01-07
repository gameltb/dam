import { render } from "@testing-library/react";
/**
 * @file MediaRenderers.test.tsx
 * @problem Visual media components (Image, Video, Markdown) require consistent rendering behavior across different node modes.
 * @requirement Verify that individual media renderer components correctly display their respective content types (images, videos, markdown).
 */
import { describe, expect, it } from "vitest";

import { ImageRenderer } from "../media/ImageRenderer";
import { MarkdownRenderer } from "../media/MarkdownRenderer";
import { VideoRenderer } from "../media/VideoRenderer";

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
