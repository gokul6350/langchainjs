import { ChatModelIntegrationTests } from "@langchain/standard-tests/vitest";
import {
  AIMessageChunk,
  HumanMessage,
} from "@langchain/core/messages";
import {
  ChatGroq,
  ChatGroqCallOptions,
  ChatGroqInput,
} from "../chat_models.js";

const TEST_IMAGE_URL =
  "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg";

class ChatGroqStandardIntegrationTests extends ChatModelIntegrationTests<
  ChatGroqCallOptions,
  AIMessageChunk,
  ChatGroqInput
> {
  constructor() {
    if (!process.env.GROQ_API_KEY) {
      throw new Error(
        "Can not run Groq integration tests because GROQ_API_KEY is not set"
      );
    }
    super({
      Cls: ChatGroq,
      chatModelHasToolCalling: true,
      chatModelHasStructuredOutput: true,
      constructorArgs: {
        model: "llama-3.3-70b-versatile",
        maxRetries: 1,
      },
      supportsStandardContentType: {
        image: ["url", "base64"],
      },
    });
  }

  async testToolMessageHistoriesListContent() {
    this.skipTestMessage(
      "testToolMessageHistoriesListContent",
      "ChatGroq",
      "Complex message types not properly implemented"
    );
  }

  async testCacheComplexMessageTypes() {
    this.skipTestMessage(
      "testCacheComplexMessageTypes",
      "ChatGroq",
      "Complex message types not properly implemented"
    );
  }

  async testStreamTokensWithToolCalls() {
    this.skipTestMessage(
      "testStreamTokensWithToolCalls",
      "ChatGroq",
      "API does not consistently call tools. TODO: re-write with better prompting for tool call."
    );
  }

  async testWithStructuredOutputIncludeRaw() {
    this.skipTestMessage(
      "testWithStructuredOutputIncludeRaw",
      "ChatGroq",
      "API does not consistently call tools. TODO: re-write with better prompting for tool call."
    );
  }

  // Override to use a vision-capable model for image tests
  async testStandardImageContentBlocks() {
    const support = (this.supportsStandardContentType?.image ?? []) as string[];
    if (!support.length) {
      this.skipTestMessage(
        "testStandardImageContentBlocks",
        "ChatGroq",
        "image not supported"
      );
      return;
    }

    const chatModel = new ChatGroq({
      model: "meta-llama/llama-4-scout-17b-16e-instruct",
      maxRetries: 1,
    });

    if (support.includes("url")) {
      const msg = new HumanMessage({
        content: [
          {
            type: "image",
            source_type: "url",
            url: TEST_IMAGE_URL,
          },
        ],
      });
      const result = await chatModel.invoke([msg]);
      this.expect(result).toBeDefined();
      this.expect(result.text).not.toBe("");
    }

    if (support.includes("base64")) {
      // Use a small base64-encoded 1x1 red PNG for the test
      const redPixelBase64 =
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwADhQGAWjR9awAAAABJRU5ErkJggg==";
      const msg = new HumanMessage({
        content: [
          {
            type: "image",
            source_type: "base64",
            data: redPixelBase64,
            mime_type: "image/png",
          },
        ],
      });
      const result = await chatModel.invoke([msg]);
      this.expect(result).toBeDefined();
      this.expect(result.text).not.toBe("");
    }
  }
}

const testClass = new ChatGroqStandardIntegrationTests();
testClass.runTests("ChatGroqStandardIntegrationTests");

