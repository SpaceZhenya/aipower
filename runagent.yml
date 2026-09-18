/**
 * browser-agent
 *
 * A minimal AI agent that can control a real Chromium browser to accomplish
 * a natural-language goal. Claude decides which browser action to take next;
 * this script executes the action with Playwright and reports back what
 * happened, in a loop, until Claude decides the task is complete.
 *
 * Usage (local):
 *   npm install
 *   npm run install-browsers
 *   export ANTHROPIC_API_KEY=sk-ant-...
 *   node agent.js "search for the weather in Tokyo and tell me today's forecast"
 *
 * Usage (GitHub Actions):
 *   See .github/workflows/run-agent.yml — runs headless, goal passed as
 *   a workflow_dispatch input, API key pulled from repo secrets.
 */

import Anthropic from "@anthropic-ai/sdk";
import { chromium } from "playwright";
import { mkdirSync, writeFileSync } from "fs";

const anthropic = new Anthropic(); // reads ANTHROPIC_API_KEY from env
const MODEL = "claude-sonnet-4-6";
const MAX_STEPS = 20; // safety cap so the agent can't loop forever

// ---------------------------------------------------------------------------
// 1. Define the tools Claude is allowed to use to control the browser.
// ---------------------------------------------------------------------------
const tools = [
  {
    name: "navigate",
    description: "Go to a URL in the browser.",
    input_schema: {
      type: "object",
      properties: { url: { type: "string" } },
      required: ["url"],
    },
  },
  {
    name: "click",
    description:
      "Click an element on the page, identified by a CSS selector or visible text.",
    input_schema: {
      type: "object",
      properties: {
        selector: {
          type: "string",
          description: "CSS selector, or exact visible text of the element.",
        },
      },
      required: ["selector"],
    },
  },
  {
    name: "type_text",
    description: "Type text into an input field identified by a CSS selector.",
    input_schema: {
      type: "object",
      properties: {
        selector: { type: "string" },
        text: { type: "string" },
        press_enter: {
          type: "boolean",
          description: "Whether to press Enter after typing.",
        },
      },
      required: ["selector", "text"],
    },
  },
  {
    name: "read_page",
    description:
      "Get a simplified text snapshot of the current page's visible content (for deciding what to do next).",
    input_schema: { type: "object", properties: {} },
  },
  {
    name: "screenshot",
    description: "Take a screenshot of the current page to visually inspect it.",
    input_schema: { type: "object", properties: {} },
  },
  {
    name: "finish",
    description: "Call this when the task is complete, with the final answer or result for the user.",
    input_schema: {
      type: "object",
      properties: { result: { type: "string" } },
      required: ["result"],
    },
  },
];

// ---------------------------------------------------------------------------
// 2. Execute a tool call against the live Playwright page.
// ---------------------------------------------------------------------------
async function executeTool(page, name, input) {
  switch (name) {
    case "navigate":
      await page.goto(input.url, { waitUntil: "domcontentloaded" });
      return { ok: true, url: page.url() };

    case "click": {
      const locator = looksLikeSelector(input.selector)
        ? page.locator(input.selector).first()
        : page.getByText(input.selector, { exact: false }).first();
      await locator.click({ timeout: 5000 });
      return { ok: true };
    }

    case "type_text": {
      const locator = looksLikeSelector(input.selector)
        ? page.locator(input.selector).first()
        : page.getByPlaceholder(input.selector).first();
      await locator.fill(input.text, { timeout: 5000 });
      if (input.press_enter) await locator.press("Enter");
      return { ok: true };
    }

    case "read_page": {
      const text = await page.evaluate(() => document.body.innerText);
      return { text: text.slice(0, 4000) }; // keep it bounded
    }

    case "screenshot": {
      const buffer = await page.screenshot({ type: "png" });
      return { image_base64: buffer.toString("base64") };
    }

    default:
      throw new Error(`Unknown tool: ${name}`);
  }
}

function looksLikeSelector(s) {
  return /^[.#\[]|^[a-z]+\[|^(div|span|a|button|input|form)\b/i.test(s);
}

// ---------------------------------------------------------------------------
// 3. The agent loop: ask Claude what to do, do it, feed the result back.
// ---------------------------------------------------------------------------
async function runAgent(goal) {
  // Headless by default (required in CI — there's no display on a GitHub
  // Actions runner). Set HEADLESS=false locally if you want to watch it work.
  const headless = process.env.HEADLESS !== "false";
  const browser = await chromium.launch({ headless });
  const page = await browser.newPage();

  const messages = [
    {
      role: "user",
      content: `Goal: ${goal}\n\nYou control a web browser via tools. Start by navigating somewhere useful, then inspect the page (read_page or screenshot) before clicking or typing, since selectors on real pages vary. Call "finish" once you have the answer.`,
    },
  ];

  for (let step = 0; step < MAX_STEPS; step++) {
    const response = await anthropic.messages.create({
      model: MODEL,
      max_tokens: 1024,
      tools,
      messages,
    });

    messages.push({ role: "assistant", content: response.content });

    const toolUse = response.content.find((b) => b.type === "tool_use");
    if (!toolUse) break; // Claude replied with plain text only; nothing left to do

    if (toolUse.name === "finish") {
      console.log("\n✅ Task complete:\n", toolUse.input.result);
      // Save a final screenshot + the result text so they're visible even
      // when running headless in CI (where there's no browser window to watch).
      try {
        mkdirSync("run-output", { recursive: true });
        const shot = await page.screenshot({ type: "png" });
        writeFileSync("run-output/final-screenshot.png", shot);
        writeFileSync("run-output/result.txt", toolUse.input.result);
      } catch (err) {
        console.warn("Could not save run output:", err.message);
      }
      break;
    }

    console.log(`\n→ Step ${step + 1}: ${toolUse.name}(${JSON.stringify(toolUse.input)})`);

    let result;
    try {
      result = await executeTool(page, toolUse.name, toolUse.input);
    } catch (err) {
      result = { error: err.message };
    }

    // Feed the result back as a tool_result so Claude can decide the next step.
    const content = result.image_base64
      ? [
          {
            type: "tool_result",
            tool_use_id: toolUse.id,
            content: [
              { type: "image", source: { type: "base64", media_type: "image/png", data: result.image_base64 } },
            ],
          },
        ]
      : [{ type: "tool_result", tool_use_id: toolUse.id, content: JSON.stringify(result) }];

    messages.push({ role: "user", content });
  }

  await browser.close();
}

const goal = process.argv.slice(2).join(" ") || "Go to wikipedia.org and tell me today's featured article title.";
runAgent(goal).catch((err) => {
  console.error("Agent failed:", err);
  process.exit(1);
});
