package ai.knowly.langtorch.capability.modality.text;

import ai.knowly.langtorch.processor.Processor;
import ai.knowly.langtorch.schema.chat.ChatMessage;
import ai.knowly.langtorch.schema.text.MultiChatMessage;
import ai.knowly.langtorch.store.memory.Memory;
import ai.knowly.langtorch.store.memory.conversation.ConversationMemoryContext;
import ai.knowly.langtorch.store.vectordb.integration.VectorStore;
import com.google.common.flogger.FluentLogger;
import com.google.common.util.concurrent.FluentFuture;
import com.google.common.util.concurrent.ListenableFuture;

import javax.inject.Inject;

import static com.google.common.util.concurrent.Futures.immediateFuture;
import static com.google.common.util.concurrent.MoreExecutors.directExecutor;

//TODO:
// This class is a copy and paste of ChatCompletionLLMCapability.
// I've added a vector store at line 32.
// Our end goal is to build an example in the empty class file 'ChatBotWithVectorStore'
// that can carry out a chat conversation with memory and respect to a vector store
// :/
/** Capability for a chat completion language model with respect to an embeddings vector store. */
public class ChatCompletionWithVectorStoreLLMCapability<I, O>
    implements TextLLMCapabilityWithMemory<
        I, MultiChatMessage, ChatMessage, O, ChatMessage, ConversationMemoryContext> {
  private static final FluentLogger logger = FluentLogger.forEnclosingClass();

  private final Processor<MultiChatMessage, ChatMessage> processor;
  private final Parsers<I, MultiChatMessage, ChatMessage, O> parsers;
  private final Memory<ChatMessage, ConversationMemoryContext> memory;
  private final VectorStore vectorStore;
  private boolean verbose;

  @Inject
  public ChatCompletionWithVectorStoreLLMCapability(
      Processor<MultiChatMessage, ChatMessage> processor,
      Parsers<I, MultiChatMessage, ChatMessage, O> parsers,
      Memory<ChatMessage, ConversationMemoryContext> memory,
      VectorStore vectorStore) {
    this.processor = processor;
    this.parsers = parsers;
    this.memory = memory;
    this.vectorStore = vectorStore;
    this.verbose = false;
  }

  protected ChatCompletionWithVectorStoreLLMCapability<I, O> withVerboseMode(boolean verbose) {
    this.verbose = verbose;
    return this;
  }

  @Override
  public MultiChatMessage preProcess(I inputData) {
    if (inputData instanceof MultiChatMessage) {
      return (MultiChatMessage) inputData;
    }
    return parsers
        .getInputParser()
        .map(parser -> parser.parse(inputData))
        .orElseThrow(
            () ->
                new IllegalArgumentException(
                    "Input data is not a MultiChatMessage and no input parser is present."));
  }

  @Override
  public O postProcess(ChatMessage outputData) {
    return parsers
        .getOutputParser()
        .map(parser -> parser.parse(outputData))
        .orElseThrow(
            () ->
                new IllegalArgumentException(
                    "Output data type is not ChatMessage and no output parser is present."));
  }

  @Override
  public Memory<ChatMessage, ConversationMemoryContext> getMemory() {
    return memory;
  }

  @Override
  public O run(I inputData) {
    return postProcess(generateMemorySideEffectResponse(preProcess(inputData)));
  }

  //TODO: not sure if we need this since its going to be duplicated from the copied class? possibly we should extend that class instead?
  private ChatMessage generateMemorySideEffectResponse(MultiChatMessage multiChatMessage) {
    if (verbose) {
      logger.atInfo().log("Memory before processing: %s", memory);
    }
    ChatMessage response = processor.run(getMessageWithMemorySideEffect(multiChatMessage));
    // Adding prompt and response.
    multiChatMessage.getMessages().forEach(memory::add);
    memory.add(response);
    return response;
  }

  //TODO: ditto
  private MultiChatMessage getMessageWithMemorySideEffect(MultiChatMessage message) {
    // Memory context being empty means that this is the first message in the conversation
    String memoryContext = memory.getMemoryContext().get();
    if (memoryContext.isEmpty()) {
      return message;
    }

    MultiChatMessage updatedMessage =
        message.getMessages().stream()
            .map(
                chatMessage ->
                    new ChatMessage(
                        String.format(
                            "%s%nBelow is my query:%n%s", memoryContext, chatMessage.toString()),
                        chatMessage.getRole(),
                        null,
                        null))
            .collect(MultiChatMessage.toMultiChatMessage());

    if (verbose) {
      logger.atInfo().log("Updated Message with Memory Side Effect: %s", updatedMessage);
    }

    return updatedMessage;
  }

  @Override
  public ListenableFuture<O> runAsync(I inputData) {
    return FluentFuture.from(immediateFuture(inputData)).transform(this::run, directExecutor());
  }
}
